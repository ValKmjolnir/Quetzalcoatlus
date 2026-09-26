#pragma once

#include "tensor/tensor.hpp"
#include "tensor/rope.hpp"
#include "gpt/attention.hpp"

namespace quetzal::gpt {

class swiglu {
private:
    tensor::tensor<float> gate_proj_pre_transposed;
    tensor::tensor<float> up_proj_pre_transposed;
    tensor::tensor<float> down_proj_pre_transposed;

public:
    swiglu(const tensor::tensor<float>& gate_proj,
           const tensor::tensor<float>& up_proj,
           const tensor::tensor<float>& down_proj) :
        gate_proj_pre_transposed(gate_proj.transpose(0, 1).contiguous()),
        up_proj_pre_transposed(up_proj.transpose(0, 1).contiguous()),
        down_proj_pre_transposed(down_proj.transpose(0, 1).contiguous()) {}
    tensor::tensor<float> forward(const tensor::tensor<float>& x) const;
};

class transformer {
private:
    tensor::tensor<float> ln1_w_;
    tensor::tensor<float> ln1_b_;
    tensor::tensor<float> ln2_w_;
    tensor::tensor<float> ln2_b_;
    swiglu ffn_;
    multi_head_attention attn_;

public:
    transformer(std::size_t d_model,
                std::size_t n_head,
                const tensor::rope<float>& rope,
                const tensor::tensor<float>& ln1_w,
                const tensor::tensor<float>& ln1_b,
                const tensor::tensor<float>& ln2_w,
                const tensor::tensor<float>& ln2_b,
                const tensor::tensor<float>& gate_proj,
                const tensor::tensor<float>& up_proj,
                const tensor::tensor<float>& down_proj,
                const tensor::tensor<float>& Wq,
                const tensor::tensor<float>& Wk,
                const tensor::tensor<float>& Wv,
                const tensor::tensor<float>& Wo) :
        ln1_w_(ln1_w), ln1_b_(ln1_b), ln2_w_(ln2_w), ln2_b_(ln2_b),
        ffn_(gate_proj, up_proj, down_proj),
        attn_(d_model, n_head, rope, Wq, Wk, Wv, Wo) {}
    tensor::tensor<float> forward(const tensor::tensor<float>& x) const;
    const auto& get_layernorm1_weight() const { return ln1_w_; }
    const auto& get_layernorm1_bias() const { return ln1_b_; }
    const auto& get_attention() const { return attn_; }
};

}
