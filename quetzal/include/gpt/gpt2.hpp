#pragma once

#include "tensor/tensor.hpp"
#include "tensor/weights_manager.hpp"
#include "gpt/transformer.hpp"

#include <vector>

namespace quetzal::gpt {

class gpt2 {
private:
    tensor::tensor<float> tok_emb;
    std::vector<transformer> blocks;
    tensor::tensor<float> ln_f_w;
    tensor::tensor<float> ln_f_b;
    tensor::tensor<float> lm_head;

public:
    gpt2(const weights_manager& wm);
    tensor::tensor<float> forward(const std::vector<std::uint32_t>& indices) const;
};

}
