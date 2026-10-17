#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

#include "tensor/tensor.hpp"
#include "util/densemap.hpp"
#include "bbpe/index_pair.hpp"
#include "model/config.hpp"

namespace quetzal::gpt {

enum class value_type {
    uint32,
    string
};

enum class weight_dtype {
    float32,
};

class weights_manager {
private:
    std::unordered_map<std::string, tensor::tensor<float>> weights_;

public:
    void read(std::ifstream&, bool);
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

class bin_reader {
private:
    util::densemap<std::string, std::uint32_t> vocab_index;
    std::vector<std::string> vocab;

    std::vector<std::string> special_vocab;

    std::vector<bbpe::index_pair> merge_pairs;

public:
    void read(std::ifstream&, bool);

    const util::densemap<std::string, std::uint32_t>& get_vocab_index() const { return vocab_index; }
    const std::vector<std::string>& get_vocab() const { return vocab; }
    const std::vector<std::string>& get_special_vocab() const { return special_vocab; }
    const std::vector<bbpe::index_pair>& get_merges() const { return merge_pairs; }
};

class qgpt_bin_reader {
private:
    weights_manager weights_manager_;
    bin_reader bin_reader_;
    model_config model_config_;

private:
    void read_config(std::ifstream&, bool);

public:
    qgpt_bin_reader(const std::string& path, bool debug_mode = false);
    const weights_manager& get_weights_manager() const { return weights_manager_; }
    const bin_reader& get_bin_reader() const { return bin_reader_; }
    const model_config& get_model_config() const { return model_config_; }
};

}
