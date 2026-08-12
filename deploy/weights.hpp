#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qgpt {

struct Tensor {
    std::string name;
    std::vector<std::uint32_t> shape;
    std::vector<float> data;  // row-major, float32
};

// Load a QGPT weights file written by deploy/export_weights.py.
// Returns tensors keyed by their state_dict name (sorted lexicographically).
std::map<std::string, Tensor> load_weights(const std::string& path);

}
