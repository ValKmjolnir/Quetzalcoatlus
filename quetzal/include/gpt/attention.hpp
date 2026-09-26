#pragma once

#include "tensor/tensor.hpp"
#include "tensor/rope.hpp"

namespace quetzal::gpt {

class multi_head_attention {
private:
    std::size_t d_model_;
    std::size_t n_head_;
    tensor::rope<float> rope_;
    tensor::tensor<float> Wq_pre_transposed;
    tensor::tensor<float> Wk_pre_transposed;
    tensor::tensor<float> Wv_pre_transposed;
    tensor::tensor<float> Wo_pre_transposed;

public:
    multi_head_attention(std::size_t d_model,
                         std::size_t n_head,
                         const tensor::rope<float>& rope,
                         const tensor::tensor<float>& Wq,
                         const tensor::tensor<float>& Wk,
                         const tensor::tensor<float>& Wv,
                         const tensor::tensor<float>& Wo) :
        d_model_(d_model), n_head_(n_head), rope_(rope),
        Wq_pre_transposed(Wq.transpose(0, 1).contiguous()),
        Wk_pre_transposed(Wk.transpose(0, 1).contiguous()),
        Wv_pre_transposed(Wv.transpose(0, 1).contiguous()),
        Wo_pre_transposed(Wo.transpose(0, 1).contiguous()) {}
    tensor::tensor<float> forward_attn(const tensor::tensor<float>& x) const;
    tensor::tensor<float> forward(const tensor::tensor<float>& x) const;
};

}
