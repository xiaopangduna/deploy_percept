#include "fixtures.hpp"

namespace percept {
namespace test {

std::vector<deploy_percept::TensorView> makeFp32TensorViews(
    const std::vector<std::vector<float>> &buffers,
    std::vector<std::vector<float>> &storage)
{
    storage = buffers;
    std::vector<deploy_percept::TensorView> views;
    views.reserve(storage.size());
    for (const auto &buf : storage)
    {
        views.push_back({buf.data(), buf.size() * sizeof(float), deploy_percept::TensorDtype::FP32});
    }
    return views;
}

std::vector<int8_t *> makeInt8PtrInputs(std::vector<std::vector<int8_t>> &buffers)
{
    std::vector<int8_t *> inputs;
    inputs.reserve(buffers.size());
    for (auto &buf : buffers)
    {
        inputs.push_back(buf.data());
    }
    return inputs;
}

} // namespace test
} // namespace percept
