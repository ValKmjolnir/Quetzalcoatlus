#pragma once

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

#include "tensor/tensor.hpp"

namespace quetzal {

enum class weight_dtype {
    float32,
};

class weights_manager {
private:
    std::unordered_map<std::string, tensor::tensor<float>> weights_;

public:
    weights_manager(const std::string& path);
    bool has(const std::string& name) const {
        return weights_.find(name) != weights_.end();
    }
    const tensor::tensor<float>& get(const std::string& name) const {
        if (!has(name)) {
            std::cerr << "[Fatal] weights_manager: no such weight: " << name << std::endl;
        }
        return weights_.at(name);
    }
    const auto& weights() const { return weights_; }
};

}
