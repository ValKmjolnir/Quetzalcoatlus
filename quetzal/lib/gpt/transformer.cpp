#include "gpt/transformer.hpp"
#include "tensor/linalg.hpp"

namespace quetzal::gpt {

tensor::tensor<float> swiglu::forward(const tensor::tensor<float>& x) const {
    auto gate = tensor::silu<float>(
        tensor::matmul_2d<float>(x, gate_proj_pre_transposed));
    auto up = tensor::matmul_2d<float>(x, up_proj_pre_transposed);
    auto down_input = tensor::mul<float>(gate, up);
    auto down = tensor::matmul_2d<float>(down_input, down_proj_pre_transposed);
    return down;
}

tensor::tensor<float> transformer::forward(const tensor::tensor<float>& x) const {
    auto res = x;
    auto attn_out = attn_.forward(tensor::layernorm<float>(res, ln1_w_, ln1_b_));
    res = tensor::add<float>(res, attn_out);
    auto ffn_out = ffn_.forward(tensor::layernorm<float>(res, ln2_w_, ln2_b_));
    res = tensor::add<float>(res, ffn_out);
    return res;
}

}
