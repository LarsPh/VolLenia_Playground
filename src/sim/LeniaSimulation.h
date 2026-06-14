#pragma once

#include "sim/DeviceVolume.h"
#include "sim/LeniaParams.h"

#include <cufft.h>

namespace vollenia {

class LeniaSimulation {
public:
    LeniaSimulation() = default;
    ~LeniaSimulation() noexcept;

    LeniaSimulation(const LeniaSimulation&) = delete;
    LeniaSimulation& operator=(const LeniaSimulation&) = delete;

    void initialize(VolumeDesc desc, const LeniaParams& params, LeniaSeedPreset seed_preset, unsigned int seed);
    void resetSeed(LeniaSeedPreset seed_preset, unsigned int seed);
    void resetImportedCells(DeviceVolumeView source_cells);
    void setParams(const LeniaParams& params);
    void rebuildKernel();
    void simulateSteps(int steps, bool validate_nan_inf);
    void destroy();

    [[nodiscard]] DeviceVolumeView currentStateView() const { return state_.view(); }
    [[nodiscard]] const LeniaParams& params() const { return params_; }
    [[nodiscard]] const LeniaStatus& status() const { return status_; }
    [[nodiscard]] bool isInitialized() const { return status_.initialized && state_.isValid(); }
    [[nodiscard]] bool kernelDirty() const { return kernel_dirty_; }

private:
    void ensureBuffers(VolumeDesc desc);
    void destroyPlans();
    void destroyNoThrow() noexcept;
    void updateStatus();

    DeviceVolume state_;
    DeviceVolume potential_;
    DeviceVolume spatial_kernel_;
    cufftComplex* state_spectrum_ = nullptr;
    cufftComplex* kernel_spectrum_ = nullptr;
    cufftComplex* potential_spectrum_ = nullptr;
    int* invalid_flag_ = nullptr;
    cufftHandle r2c_plan_ = 0;
    cufftHandle c2r_plan_ = 0;
    VolumeDesc desc_;
    LeniaParams params_;
    LeniaStatus status_;
    std::size_t spectrum_count_ = 0;
    unsigned int seed_ = 1;
    LeniaSeedPreset seed_preset_ = LeniaSeedPreset::ReferenceRandomBox;
    bool buffers_ready_ = false;
    bool kernel_dirty_ = true;
};

} // namespace vollenia
