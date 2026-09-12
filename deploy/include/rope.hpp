#pragma once

#include <cmath>

#include "omp.hpp"
#include "tensor.hpp"
#include "assert.hpp"

namespace quetzal::tensor {

template<typename T>
class rope {
private:
    std::size_t seq_;
    std::size_t d_k_;
    tensor<T> freqs_;
    tensor<T> cos_;
    tensor<T> sin_;

public:
    rope(std::size_t seq, std::size_t d_k):
        seq_(seq), d_k_(d_k),
        freqs_({d_k / 2}), cos_({seq, d_k / 2}), sin_({seq, d_k / 2}) {
        OMP_FOR
        for (std::size_t i = 0; i < d_k / 2; ++i) {
            freqs_.data()[i] = T(1.0) / std::pow(T(10000.0), T(2) * i / T(d_k));
        }
        OMP_FOR
        for (std::size_t i = 0; i < seq; ++i) {
            for (std::size_t j = 0; j < d_k / 2; ++j) {
                cos_.data()[i * d_k / 2 + j] = std::cos(freqs_.data()[j] * i);
                sin_.data()[i * d_k / 2 + j] = std::sin(freqs_.data()[j] * i);
            }
        }
    }

    void apply(tensor<T>& x) const {
        QUETZAL_ASSERT(x.is_contiguous(), "[rope] tensor must be contiguous");
        QUETZAL_ASSERT(x.shape().size() >= 2, "[rope] shape mismatch");
        const std::size_t d_k = x.shape()[x.shape().size() - 1];
        const std::size_t seq = x.shape()[x.shape().size() - 2];
        QUETZAL_ASSERT(seq == seq_, "[rope] shape mismatch: seq");
        QUETZAL_ASSERT(d_k == d_k_, "[rope] shape mismatch: d_k");

        const std::size_t total = x.total_size();
        OMP_FOR
        for (std::size_t i = 0; i < total; i += seq * d_k) {
            for (std::size_t s = 0; s < seq; ++s) {
                for (std::size_t j = 0; j < d_k / 2; ++j) {
                    std::size_t offset = i + s * d_k;
                    T even = x.data()[offset + j * 2];
                    T odd  = x.data()[offset + j * 2 + 1];
                    T cos_val = cos_.data()[s * d_k / 2 + j];
                    T sin_val = sin_.data()[s * d_k / 2 + j];
                    x.data()[offset + j * 2]     = even * cos_val - odd * sin_val;
                    x.data()[offset + j * 2 + 1] = even * sin_val + odd * cos_val;
                }
            }
        }
    }
};

}
