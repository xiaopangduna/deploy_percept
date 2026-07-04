#pragma once

#include "deploy_percept/common/types.hpp"

#include <cstdint>
#include <vector>

namespace percept {
namespace test {

std::vector<deploy_percept::TensorView> makeFp32TensorViews(
    const std::vector<std::vector<float>> &buffers,
    std::vector<std::vector<float>> &storage);

std::vector<int8_t *> makeInt8PtrInputs(std::vector<std::vector<int8_t>> &buffers);

} // namespace test
} // namespace percept
