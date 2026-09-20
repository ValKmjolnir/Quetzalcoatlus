#include "gpt/attention.hpp"
#include "tensor/linalg.hpp"
#include "tensor/rope.hpp"

#include <cmath>

namespace quetzal::gpt {

tensor::tensor<float>
multi_head_attention::forward(const tensor::tensor<float>& x) const {
    // pytorch nn.Linear stores weights in (out_dim, in_dim)
    // but in forward progress, it still uses x @ (in_dim, out_dim)
    // so we need to transpose them here too

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_.transpose(0, 1));
    auto K = tensor::matmul_2d<float>(x, Wk_.transpose(0, 1));
    auto V = tensor::matmul_2d<float>(x, Wv_.transpose(0, 1));

    auto seq_len = x.shape()[0];
    auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1);

    tensor::rope<float> rope(seq_len, d_k);
    rope.apply(Q);
    rope.apply(K);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::div<float>(tensor::matmul_batch<float>(Q, K.transpose(1, 2)), std::sqrt(d_k));
    tensor::apply_causal_mask<float>(scores);

    auto attn = tensor::softmax<float>(scores);
    // (n_head, seq_len, seq_len) @ (n_head, seq_len, d_k) -> (n_head, seq_len, d_k)
    auto out = tensor::matmul_batch<float>(attn, V);

    // (n_head, seq_len, d_k) -> (seq_len, n_head, d_k)
    out = out.transpose(0, 1).contiguous();
    // (seq_len, n_head, d_k) -> (seq_len, d_model)
    out = out.reshape({seq_len, d_model_});
    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    out = tensor::matmul_2d<float>(out, Wo_.transpose(0, 1));
    return out;
}

}
