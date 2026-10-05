#pragma once

#include "tensor/tensor.hpp"
#include "tensor/rope.hpp"
#include "util/perf_info.hpp"

#include <cstring>

namespace quetzal::gpt {

class multi_head_attention {
private:
    std::size_t d_model_;
    std::size_t n_head_;
    std::size_t max_seq_len_;
    tensor::rope<float> rope_;
    tensor::tensor<float> Wq_pre_transposed;
    tensor::tensor<float> Wk_pre_transposed;
    tensor::tensor<float> Wv_pre_transposed;
    tensor::tensor<float> Wo_pre_transposed;
    tensor::tensor<float> K_cache;
    tensor::tensor<float> V_cache;
    std::size_t cache_len;

private:
    tensor::tensor<float> get_real_K() const {
        auto d_k = d_model_ / n_head_;
        tensor::tensor<float> res({n_head_, cache_len, d_k});
        for (std::size_t h = 0; h < n_head_; ++h) {
            std::memcpy(res.data() + h * cache_len * d_k,
                        K_cache.data() + h * max_seq_len_ * d_k,
                        cache_len * d_k * sizeof(float));
        }
        return res;
    }

    tensor::tensor<float> get_real_V() const {
        auto d_k = d_model_ / n_head_;
        tensor::tensor<float> res({n_head_, cache_len, d_k});
        for (std::size_t h = 0; h < n_head_; ++h) {
            std::memcpy(res.data() + h * cache_len * d_k,
                        V_cache.data() + h * max_seq_len_ * d_k,
                        cache_len * d_k * sizeof(float));
        }
        return res;
    }

public:
    multi_head_attention(const std::size_t d_model,
                         const std::size_t n_head,
                         const std::size_t max_seq_len,
                         const tensor::rope<float>& rope,
                         const tensor::tensor<float>& Wq,
                         const tensor::tensor<float>& Wk,
                         const tensor::tensor<float>& Wv,
                         const tensor::tensor<float>& Wo) :
        d_model_(d_model), n_head_(n_head), max_seq_len_(max_seq_len),
        rope_(rope),
        Wq_pre_transposed(Wq.transpose(0, 1).contiguous()),
        Wk_pre_transposed(Wk.transpose(0, 1).contiguous()),
        Wv_pre_transposed(Wv.transpose(0, 1).contiguous()),
        Wo_pre_transposed(Wo.transpose(0, 1).contiguous()),
        K_cache({n_head, max_seq_len, d_model / n_head}),
        V_cache({n_head, max_seq_len, d_model / n_head}), cache_len(0) {}
    void reset() {
        std::memset(K_cache.data(), 0, K_cache.total_size() * sizeof(float));
        std::memset(V_cache.data(), 0, V_cache.total_size() * sizeof(float));
        cache_len = 0;
    }
    tensor::tensor<float> forward_attn(const tensor::tensor<float>& x) const;
    tensor::tensor<float> forward(const tensor::tensor<float>& x) const;
    tensor::tensor<float> prefill(const tensor::tensor<float>& x);
    tensor::tensor<float> decode(const tensor::tensor<float>& x);
    tensor::tensor<float> decode_perf(const tensor::tensor<float>& x,
                                      util::attn_perf_info& api);
};

}
