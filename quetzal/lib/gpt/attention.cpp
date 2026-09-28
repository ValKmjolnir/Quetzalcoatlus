#include "gpt/attention.hpp"
#include "tensor/linalg.hpp"
#include "tensor/rope.hpp"

#include <cmath>
#include <chrono>

namespace quetzal::gpt {

tensor::tensor<float>
multi_head_attention::forward_attn(const tensor::tensor<float>& x) const {
    // only calculate attention matrix, for visualization

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);

    auto seq_len = x.shape()[0];
    auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2)) / std::sqrt(d_k);
    return scores;
}

tensor::tensor<float>
multi_head_attention::forward(const tensor::tensor<float>& x) const {
    // pytorch nn.Linear stores weights in (out_dim, in_dim)
    // but in forward progress, it still uses x @ (in_dim, out_dim)
    // so we need to transpose them here too

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    auto seq_len = x.shape()[0];
    auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2).contiguous())
                / std::sqrt(d_k);
    tensor::apply_causal_mask<float>(scores);

    auto attn = tensor::softmax<float>(scores);
    // (n_head, seq_len, seq_len) @ (n_head, seq_len, d_k) -> (n_head, seq_len, d_k)
    auto out = tensor::matmul_batch<float>(attn, V);

    // (n_head, seq_len, d_k) -> (seq_len, n_head, d_k)
    out = out.transpose(0, 1).contiguous();
    // (seq_len, n_head, d_k) -> (seq_len, d_model)
    out = out.reshape({seq_len, d_model_});
    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    out = tensor::matmul_2d<float>(out, Wo_pre_transposed);
    return out;
}

tensor::tensor<float>
multi_head_attention::forward_perf(const tensor::tensor<float>& x,
                                   util::attn_perf_info& api) const {
    using clk = std::chrono::high_resolution_clock;

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto start = clk::now();
    auto total_start = start;
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    auto seq_len = x.shape()[0];
    auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);
    auto end = clk::now();
    api.QKV_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    start = clk::now();
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2).contiguous())
                / std::sqrt(d_k);
    tensor::apply_causal_mask<float>(scores);
    end = clk::now();
    api.QK_score_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = clk::now();
    auto attn = tensor::softmax<float>(scores);
    // (n_head, seq_len, seq_len) @ (n_head, seq_len, d_k) -> (n_head, seq_len, d_k)
    auto out = tensor::matmul_batch<float>(attn, V);

    // (n_head, seq_len, d_k) -> (seq_len, n_head, d_k)
    out = out.transpose(0, 1).contiguous();
    // (seq_len, n_head, d_k) -> (seq_len, d_model)
    out = out.reshape({seq_len, d_model_});
    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    out = tensor::matmul_2d<float>(out, Wo_pre_transposed);
    end = clk::now();
    api.merge_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    auto total_end = clk::now();
    api.total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    return out;
}

}
