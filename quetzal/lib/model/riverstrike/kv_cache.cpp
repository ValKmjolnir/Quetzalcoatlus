#include "model/riverstrike/kv_cache.hpp"
#include "tensor/assert.hpp"

namespace quetzal::gpt {

void mha_kv_cache::prefill(const tensor::tensor<float>& K,
                           const tensor::tensor<float>& V,
                           const std::size_t seq_len) {
    QUETZAL_ASSERT(seq_len <= max_seq_len_, "[mha_kv_cache] sequence length out of range");
    QUETZAL_ASSERT(K.is_contiguous(), "[mha_kv_cache] K must be contiguous");
    QUETZAL_ASSERT(V.is_contiguous(), "[mha_kv_cache] V must be contiguous");

    const auto d_k = d_model_ / n_head_;
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
}

void mha_kv_cache::update(const tensor::tensor<float>& K,
                          const tensor::tensor<float>& V) {
    QUETZAL_ASSERT(cache_len < max_seq_len_, "[mha_kv_cache] sequence length out of range");
    const auto d_k = d_model_ / n_head_;
    for (std::size_t h = 0; h < n_head_; ++h) {
        std::memcpy(K_cache.data() + (h * max_seq_len_ + cache_len) * d_k,
                    K.data()       + h * d_k,
                    d_k * sizeof(float));
        std::memcpy(V_cache.data() + (h * max_seq_len_ + cache_len) * d_k,
                    V.data()       + h * d_k,
                    d_k * sizeof(float));
    }
    ++cache_len;
}

}
