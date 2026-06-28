#pragma once

#include "model/ModelSpec.h"
#include "sim/DeviceMultiVolume.h"
#include "sim/DeviceVolume.h"
#include "sim/KernelBank.h"

#include <cufft.h>

#include <cstddef>
#include <string>
#include <vector>

namespace vollenia {

struct ExpandedFlowStatus {
    bool initialized = false;
    bool playing = true;
    bool last_step_had_invalid_values = false;
    VolumeDesc desc;
    std::size_t byte_size = 0;
    unsigned long long generation = 0;
    int channel_count = 0;
    int kernel_count = 0;
    int render_channel = 0;
    double mass_before = 0.0;
    double mass_after = 0.0;
    double mass_ratio = 1.0;
    const char* update_mode = "unknown";
    std::string model_name;
    std::string status = "Not initialized";
    std::string last_error;
};

class ExpandedFlowSimulation {
public:
    ExpandedFlowSimulation() = default;
    ~ExpandedFlowSimulation() noexcept;

    ExpandedFlowSimulation(const ExpandedFlowSimulation&) = delete;
    ExpandedFlowSimulation& operator=(const ExpandedFlowSimulation&) = delete;

    void initialize(VolumeDesc desc, const ModelSpec& spec);
    void resetSeed(unsigned int seed);
    void simulateSteps(int steps);
    void setRenderChannel(int channel);
    void destroy();

    [[nodiscard]] DeviceVolumeView currentRenderView() const;
    [[nodiscard]] DeviceVolumeView channelView(int channel) const;
    [[nodiscard]] const ExpandedFlowStatus& status() const { return status_; }
    [[nodiscard]] const ModelSpec& spec() const { return spec_; }
    [[nodiscard]] int renderChannel() const { return render_channel_; }
    [[nodiscard]] int channelCount() const { return channels_; }
    [[nodiscard]] bool isInitialized() const { return status_.initialized && state_.isValid(); }

private:
    void ensureBuffers(VolumeDesc desc, int channels);
    void destroyPlans();
    void destroyNoThrow() noexcept;
    void computeAffinity();
    void stepExpandedAdditive();
    void stepFlow();
    void updateStatus(const char* status_text);
    [[nodiscard]] double matterMass() const;

    ModelSpec spec_;
    DeviceMultiVolume state_;
    DeviceMultiVolume next_;
    DeviceMultiVolume affinity_;
    KernelBank kernel_bank_;
    DeviceVolume potential_;
    DeviceVolume matter_sum_;
    cufftComplex* state_spectra_ = nullptr;
    cufftComplex* work_spectrum_ = nullptr;
    float3* grad_u_ = nullptr;
    float3* grad_a_sum_ = nullptr;
    float3* flow_ = nullptr;
    int* invalid_flag_ = nullptr;
    cufftHandle r2c_plan_ = 0;
    cufftHandle c2r_plan_ = 0;
    VolumeDesc desc_;
    std::size_t voxel_count_ = 0;
    std::size_t spectrum_count_ = 0;
    int channels_ = 0;
    int render_channel_ = 0;
    unsigned int seed_ = 1;
    ExpandedFlowStatus status_;
};

} // namespace vollenia
