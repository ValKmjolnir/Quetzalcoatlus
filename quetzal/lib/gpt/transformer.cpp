#include "gpt/transformer.hpp"
#include "tensor/linalg.hpp"

#include <chrono>

namespace quetzal::gpt {

tensor::tensor<float> swiglu::forward(const tensor::tensor<float>& x) const {
    auto gate = tensor::silu<float>(
        tensor::matmul_2d<float>(x, gate_proj_pre_transposed));
    auto up = tensor::matmul_2d<float>(x, up_proj_pre_transposed);
    auto down_input = gate * up;
    auto down = tensor::matmul_2d<float>(down_input, down_proj_pre_transposed);
    return down;
}

tensor::tensor<float> transformer::forward(const tensor::tensor<float>& x) const {
    auto res = x;
    auto attn_out = attn_.forward(tensor::layernorm<float>(res, ln1_w_, ln1_b_));
    res = res + attn_out;
    auto ffn_out = ffn_.forward(tensor::layernorm<float>(res, ln2_w_, ln2_b_));
    res += ffn_out;
    return res;
}

tensor::tensor<float> transformer::forward_perf(const tensor::tensor<float>& x,
                                                util::transformer_perf_info& tpi) const {
    using clk = std::chrono::high_resolution_clock;

    auto total_begin = clk::now();

    auto res = x;
    auto attn_out = attn_.forward_perf(tensor::layernorm<float>(res, ln1_w_, ln1_b_), tpi.attn_time);

    res = res + attn_out;

    auto start = clk::now();
    auto ffn_out = ffn_.forward(tensor::layernorm<float>(res, ln2_w_, ln2_b_));
    auto end = clk::now();
    tpi.ffn_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    res += ffn_out;

    auto total_end = clk::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count();
    tpi.total_time = dur / 1000.f;

    return res;
}

}
