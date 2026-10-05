#include "gpt/attention.hpp"
#include "tensor/linalg.hpp"
#include "tensor/rope.hpp"

#include <cmath>
#include <chrono>

namespace quetzal::gpt {

tensor::tensor<float>
multi_head_attention::forward_attn(const tensor::tensor<float>& x) const {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
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
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
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
multi_head_attention::prefill(const tensor::tensor<float>& x) {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    // pytorch nn.Linear stores weights in (out_dim, in_dim)
    // but in forward progress, it still uses x @ (in_dim, out_dim)
    // so we need to transpose them here too

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    auto seq_len = x.shape()[0];
    QUETZAL_ASSERT(seq_len <= max_seq_len_, "[mha] sequence length out of range");
    auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);

    reset();
    for (std::size_t h = 0; h < n_head_; ++h) {
        std::memcpy(K_cache.data() + h * max_seq_len_ * d_k,
                    K.data()       + h * seq_len      * d_k,
                    seq_len * d_k * sizeof(float));
        std::memcpy(V_cache.data() + h * max_seq_len_ * d_k,
                    V.data()       + h * seq_len      * d_k,
                    seq_len * d_k * sizeof(float));
    }
    cache_len = seq_len;

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
multi_head_attention::decode(const tensor::tensor<float>& x) {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    QUETZAL_ASSERT(x.shape()[0] == 1, "[mha] shape mismatch");
    // pytorch nn.Linear stores weights in (out_dim, in_dim)
    // but in forward progress, it still uses x @ (in_dim, out_dim)
    // so we need to transpose them here too

    // (1, d_model) @ (d_model, d_model) -> (1, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    auto seq_len = x.shape()[0];
    auto d_k = d_model_ / n_head_;

    // (1, d_model) -> (1, n_head, d_k) -> (n_head, 1, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q, cache_len);
    rope_.apply(K, cache_len);

    for (std::size_t h = 0; h < n_head_; ++h) {
        std::memcpy(K_cache.data() + (h * max_seq_len_ + cache_len) * d_k,
                    K.data()       + h * d_k,
                    d_k * sizeof(float));
        std::memcpy(V_cache.data() + (h * max_seq_len_ + cache_len) * d_k,
                    V.data()       + h * d_k,
                    d_k * sizeof(float));
    }
    ++cache_len;

    K = get_real_K();
    V = get_real_V();

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2).contiguous())
                / std::sqrt(d_k);

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
multi_head_attention::decode_perf(const tensor::tensor<float>& x,
                                  util::attn_perf_info& api) {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    QUETZAL_ASSERT(x.shape()[0] == 1, "[mha] shape mismatch");
    using clk = std::chrono::high_resolution_clock;

    // (1, d_model) @ (d_model, d_model) -> (1, d_model)
    auto start = clk::now();
    auto total_start = start;
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    auto seq_len = x.shape()[0];
    auto d_k = d_model_ / n_head_;

    // (1, d_model) -> (1, n_head, d_k) -> (n_head, 1, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q, cache_len);
    rope_.apply(K, cache_len);
    auto end = clk::now();
    api.QKV_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    for (std::size_t h = 0; h < n_head_; ++h) {
        std::memcpy(K_cache.data() + (h * max_seq_len_ + cache_len) * d_k,
                    K.data()       + h * d_k,
                    d_k * sizeof(float));
        std::memcpy(V_cache.data() + (h * max_seq_len_ + cache_len) * d_k,
                    V.data()       + h * d_k,
                    d_k * sizeof(float));
    }
    ++cache_len;

    K = get_real_K();
    V = get_real_V();

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    start = clk::now();
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2).contiguous())
                / std::sqrt(d_k);
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
