#include "render/CudaVolumeRenderer.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace vollenia {

namespace {

__device__ float3 add3(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 mul3(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float length3(float3 a) { return sqrtf(dot3(a, a)); }
__device__ float3 normalize3(float3 a)
{
    const float len = fmaxf(length3(a), 1.0e-6f);
    return mul3(a, 1.0f / len);
}

__device__ float clamp01(float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ unsigned char toByte(float value)
{
    return static_cast<unsigned char>(255.0f * clamp01(value) + 0.5f);
}

__device__ bool intersectBox(float3 origin, float3 direction, float3 box_min, float3 box_max, float& t_near, float& t_far)
{
    const float3 inv_dir = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    const float3 t0 = make_float3(
        (box_min.x - origin.x) * inv_dir.x,
        (box_min.y - origin.y) * inv_dir.y,
        (box_min.z - origin.z) * inv_dir.z);
    const float3 t1 = make_float3(
        (box_max.x - origin.x) * inv_dir.x,
        (box_max.y - origin.y) * inv_dir.y,
        (box_max.z - origin.z) * inv_dir.z);

    const float3 t_min = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    const float3 t_max = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    t_near = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    t_far = fminf(fminf(t_max.x, t_max.y), t_max.z);
    return t_far >= fmaxf(t_near, 0.0f);
}

__device__ float3 transferColor(float density)
{
    const float d = clamp01(density);
    const float3 low = make_float3(0.08f, 0.22f, 0.75f);
    const float3 mid = make_float3(0.20f, 0.88f, 0.78f);
    const float3 high = make_float3(1.0f, 0.70f, 0.20f);

    if (d < 0.5f) {
        const float t = d * 2.0f;
        return make_float3(
            low.x + (mid.x - low.x) * t,
            low.y + (mid.y - low.y) * t,
            low.z + (mid.z - low.z) * t);
    }

    const float t = (d - 0.5f) * 2.0f;
    return make_float3(
        mid.x + (high.x - mid.x) * t,
        mid.y + (high.y - mid.y) * t,
        mid.z + (high.z - mid.z) * t);
}

__global__ void raymarchKernel(
    uchar4* output,
    int width,
    int height,
    cudaTextureObject_t volume_texture,
    CameraFrame camera,
    RenderParams params)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float ndc_x = (2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 1.0f);
    // map PBO's 0 (top) to NDC 1 (top), 1 (bottom) to NDC -1 (bottom)
    const float ndc_y = (1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
    const float tan_half_fov = tanf(camera.fov_y_degrees * 0.00872664626f);
    const float3 direction = normalize3(add3(
        camera.forward,
        add3(mul3(camera.right, ndc_x * camera.aspect * tan_half_fov), mul3(camera.up, ndc_y * tan_half_fov))));

    float t_near = 0.0f;
    float t_far = 0.0f;
    uchar4 out = make_uchar4(5, 9, 14, 255);
    if (!intersectBox(camera.position, direction, make_float3(-1.0f, -1.0f, -1.0f), make_float3(1.0f, 1.0f, 1.0f), t_near, t_far)) {
        output[y * width + x] = out;
        return;
    }

    float t = fmaxf(t_near, 0.0f);
    const float step = fmaxf(params.step_size, 0.001f);
    const int max_steps = max(params.max_steps, 1);
    const float threshold = fmaxf(params.threshold, 0.0f);

    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float max_density = 0.0f;
    bool first_hit = false;
    float first_density = 0.0f;

    for (int i = 0; i < max_steps && t <= t_far; ++i) {
        const float3 pos = add3(camera.position, mul3(direction, t));
        const float sample = tex3D<float>(
            volume_texture,
            pos.x * 0.5f + 0.5f,
            pos.y * 0.5f + 0.5f,
            pos.z * 0.5f + 0.5f);
        const float density = fmaxf(sample - threshold, 0.0f);

        if (params.mode == VolumeRenderMode::MIP) {
            max_density = fmaxf(max_density, sample);
        } else if (params.mode == VolumeRenderMode::FirstHit) {
            if (sample > threshold) {
                first_hit = true;
                first_density = sample;
                break;
            }
        } else if (density > 0.0f) {
            const float alpha = 1.0f - expf(-params.density_scale * density * step);
            const float3 color = transferColor(sample);
            accum = add3(accum, mul3(color, transmittance * alpha));
            transmittance *= (1.0f - alpha);
            if (transmittance < params.early_exit_transmittance) {
                break;
            }
        }

        t += step;
    }

    if (params.mode == VolumeRenderMode::MIP) {
        const float3 color = transferColor(max_density);
        accum = mul3(color, params.brightness * max_density);
    } else if (params.mode == VolumeRenderMode::FirstHit) {
        if (first_hit) {
            const float3 color = transferColor(first_density);
            accum = mul3(color, params.brightness);
        } else {
            accum = make_float3(0.0f, 0.0f, 0.0f);
        }
    } else {
        accum = mul3(accum, params.brightness);
    }

    out.x = toByte(accum.x);
    out.y = toByte(accum.y);
    out.z = toByte(accum.z);
    out.w = 255;
    output[y * width + x] = out;
}

} // namespace

CudaVolumeRenderer::~CudaVolumeRenderer() noexcept
{
    destroyNoThrow();
}

void CudaVolumeRenderer::uploadVolume(DeviceVolumeView volume)
{
    if (volume.data == nullptr || !isValidVolumeDesc(volume.desc)) {
        throw std::runtime_error("Cannot upload invalid volume to CUDA renderer");
    }

    if (!hasTexture() || !volumeDescEquals(volume_desc_, volume.desc)) {
        createTextureStorage(volume.desc);
    }

    const cudaExtent extent = make_cudaExtent(
        static_cast<std::size_t>(volume_desc_.nx),
        static_cast<std::size_t>(volume_desc_.ny),
        static_cast<std::size_t>(volume_desc_.nz));

    cudaMemcpy3DParms copy_params {};
    copy_params.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(volume.data),
        static_cast<std::size_t>(volume_desc_.nx) * sizeof(float),
        static_cast<std::size_t>(volume_desc_.nx),
        static_cast<std::size_t>(volume_desc_.ny));
    copy_params.dstArray = array_;
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyDeviceToDevice;
    VOL_CUDA_CHECK(cudaMemcpy3D(&copy_params));
}

void CudaVolumeRenderer::createTextureStorage(VolumeDesc desc)
{
    destroy();
    volume_desc_ = desc;

    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    const cudaExtent extent = make_cudaExtent(
        static_cast<std::size_t>(volume_desc_.nx),
        static_cast<std::size_t>(volume_desc_.ny),
        static_cast<std::size_t>(volume_desc_.nz));
    VOL_CUDA_CHECK(cudaMalloc3DArray(&array_, &channel_desc, extent));

    cudaResourceDesc resource_desc {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = array_;

    cudaTextureDesc texture_desc {};
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 1;

    VOL_CUDA_CHECK(cudaCreateTextureObject(&texture_, &resource_desc, &texture_desc, nullptr));
}

void CudaVolumeRenderer::render(PboMapping output, int width, int height, const CameraFrame& camera, const RenderParams& params)
{
    if (output.device_ptr == nullptr || width <= 0 || height <= 0) {
        return;
    }
    if (!hasTexture()) {
        throw std::runtime_error("CUDA volume renderer has no texture object");
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1U) / block.y);
    raymarchKernel<<<grid, block>>>(output.device_ptr, width, height, texture_, camera, params);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void CudaVolumeRenderer::destroy()
{
    if (texture_ != 0) {
        VOL_CUDA_CHECK(cudaDestroyTextureObject(texture_));
        texture_ = 0;
    }
    if (array_ != nullptr) {
        VOL_CUDA_CHECK(cudaFreeArray(array_));
        array_ = nullptr;
    }
    volume_desc_ = {};
}

void CudaVolumeRenderer::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy CUDA volume renderer cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
