#pragma once

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace quetzal {

enum class weight_dtype {
    float32,
};

struct weights {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> data;

    std::size_t numel() const {
        std::size_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        return n;
    }

    tensor::tensor<float> to_tensor() const {
        return tensor::tensor<float>(shape, data);
    }
};

std::unordered_map<std::string, weights> load_weights(const std::string& path);

}
