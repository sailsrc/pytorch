#pragma once

#include <c10/core/TensorOptions.h>

// device_lazy_init() is always compiled, even for CPU-only builds.

namespace torch {
namespace utils {

/**
 * This mechanism of lazy initialization is designed for each device backend.
 * Currently, CUDA and XPU follow this design. This function `device_lazy_init`
 * MUST be called before you attempt to access any Type(CUDA or XPU) object
 * from ATen, in any way. It guarantees that the device runtime status is lazily
 * initialized when the first runtime API is requested.
 *
 * Here are some common ways that a device object may be retrieved:
 *   - You call getNonVariableType or getNonVariableTypeOpt
 *   - You call toBackend() on a Type
 *
 * It's important to do this correctly, because if you forget to add it you'll
 * get an oblique error message seems like "Cannot initialize CUDA without
 * ATen_cuda library" if you try to use CUDA functionality from a CPU-only
 * build, which is not good UX.
 */
void device_lazy_init(at::DeviceType device_type);
void set_requires_device_init(at::DeviceType device_type, bool value);

static void maybe_initialize_device(at::Device& device) {
  if (device.is_cuda() || device.is_xpu()) {
    device_lazy_init(device.type());
  }
}

static void maybe_initialize_device(const at::TensorOptions& options) {
  if (options.device().is_cuda() || options.device().is_xpu()) {
    torch::utils::device_lazy_init(options.device().type());
  }
}

} // namespace utils
} // namespace torch
