#pragma once

#include "tensor/omp.hpp"
#include "tensor/tensor.hpp"
#include "tensor/assert.hpp"
#include "tensor/linalg.hpp"

#include <cstring>
#include <limits>

namespace quetzal::tensor {

template<typename T>
tensor<T> silu_mul_fusion(const tensor<T>& a, const tensor<T>& b) {
    QUETZAL_ASSERT(a.is_contiguous(), "[silu_mul_fusion] tensors must be contiguous");
    QUETZAL_ASSERT(b.is_contiguous(), "[silu_mul_fusion] tensors must be contiguous");
    QUETZAL_ASSERT(tensor<T>::shape_equal(a, b), "[silu_mul_fusion] shape mismatch");

    idx_t n = a.total_size();
    tensor<T> res(a.shape());

    OMP_FOR_IF(n >= omp_min_macs)
    for (idx_t i = 0; i < n; ++i) {
        const T x = a.data()[i];
        res.data()[i] = x * (T(1) / (T(1) + std::exp(-x))) * b.data()[i];
    }

    return res;
}

template<typename T>
tensor<T> causal_mask_softmax_fusion(const tensor<T>& a) {
    QUETZAL_ASSERT(a.is_contiguous(), "[softmax_causal] tensors must be contiguous");
    QUETZAL_ASSERT(a.shape().size() >= 2, "[softmax_causal] shape mismatch");

    const idx_t last_dim = a.shape()[a.shape().size() - 1];
    const idx_t second_last_dim = a.shape()[a.shape().size() - 2];
    QUETZAL_ASSERT(last_dim == second_last_dim, "[softmax_causal] shape mismatch");

    const idx_t n = a.total_size();
    const idx_t seq = a.shape().back();
    const idx_t n_head = n / seq / seq;

    tensor<T> b(a.shape());
    std::memset(b.data(), 0, b.total_size() * sizeof(T));

    OMP_FOR_IF(n_head > 1 && n >= omp_min_elems)
    for (idx_t h = 0; h < n_head; ++h) {
        const T* A = a.data() + h * seq * seq;
        T*       B = b.data() + h * seq * seq;
        for (idx_t i = 0; i < seq; ++i) {
            T max_num = -std::numeric_limits<T>::infinity();
            for (idx_t j = 0; j <= i; ++j) {
                max_num = (std::max)(max_num, A[i * seq + j]);
            }

            T sum = 0;
            for (idx_t j = 0; j <= i; ++j) {
                T tmp = std::exp(A[i * seq + j] - max_num);
                sum += tmp;
                B[i * seq + j] = tmp;
            }
            for (idx_t j = 0; j <= i; ++j) {
                B[i * seq + j] /= sum;
            }
        }
    }

    return b;
}

}
