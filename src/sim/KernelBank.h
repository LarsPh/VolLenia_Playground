#pragma once

#include "core/VolumeDesc.h"
#include "model/ModelSpec.h"

#include <cufft.h>

#include <filesystem>
#include <vector>

namespace vollenia {

class KernelBank {
public:
    KernelBank() = default;
    ~KernelBank() noexcept;

    KernelBank(const KernelBank&) = delete;
    KernelBank& operator=(const KernelBank&) = delete;

    void build(VolumeDesc desc, const ModelSpec& spec, const std::string& debug_name);
    void destroy();

    [[nodiscard]] const cufftComplex* spectrum(int kernel_index) const;
    [[nodiscard]] int kernelCount() const { return kernel_count_; }
    [[nodiscard]] std::size_t spectrumCount() const { return spectrum_count_; }
    [[nodiscard]] bool isValid() const { return spectra_ != nullptr && kernel_count_ > 0 && spectrum_count_ > 0; }

private:
    void destroyNoThrow() noexcept;
    void dumpKernelDebug(
        const std::filesystem::path& output_dir,
        const std::string& debug_name,
        const KernelSpec& kernel,
        VolumeDesc desc,
        const std::vector<float>& fft_origin_kernel) const;

    cufftComplex* spectra_ = nullptr;
    cufftHandle r2c_plan_ = 0;
    VolumeDesc desc_;
    int kernel_count_ = 0;
    std::size_t spectrum_count_ = 0;
};

} // namespace vollenia
