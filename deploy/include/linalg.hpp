#include "include/omp.hpp"
#include "include/tensor.hpp"

#include <cassert>
#include <cmath>

namespace quetzal::tensor {

template<typename T>
bool shape_equal(const tensor<T>& a, const tensor<T>& b) {
    return a.shape() == b.shape();
}

template<typename T>
tensor<T> add(const tensor<T>& a, const tensor<T>& b) {
    assert(shape_equal(a, b) && "[add] shape mismatch");
    assert(a.is_contiguous() && b.is_contiguous() && "[add] tensors must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> c(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        c.data()[i] = a.data()[i] + b.data()[i];
    }

    return c;
}

template<typename T>
tensor<T> mul(const tensor<T>& a, const tensor<T>& b) {
    assert(shape_equal(a, b) && "[mul] shape mismatch");
    assert(a.is_contiguous() && b.is_contiguous() && "[mul] tensors must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> c(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        c.data()[i] = a.data()[i] * b.data()[i];
    }

    return c;
}

template<typename T>
tensor<T> silu(const tensor<T>& a) {
    assert(a.is_contiguous() && "[silu] tensors must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> b(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        const T x = a.data()[i];
        b.data()[i] = x * (T(1) / (T(1) + std::exp(-x)));
    }

    return b;
}

template<typename T>
tensor<T> sigmoid(const tensor<T>& a) {
    assert(a.is_contiguous() && "[sigmoid] tensors must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> b(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        b.data()[i] = T(1) / (T(1) + std::exp(-a.data()[i]));
    }

    return b;
}

}