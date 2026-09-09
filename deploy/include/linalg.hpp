#include "omp.hpp"
#include "tensor.hpp"

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
tensor<T> mul(const tensor<T>& a, T b) {
    assert(a.is_contiguous() && "[mul] tensor must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> c(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        c.data()[i] = a.data()[i] * b;
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

// only see last dimension
template<typename T>
tensor<T> softmax(const tensor<T>& a) {
    assert(a.is_contiguous() && "[softmax] tensors must be contiguous");

    std::size_t n = a.total_size();
    std::size_t n_last = a.shape().back();

    tensor<T> b(a.shape());
    OMP_FOR
    for (std::size_t i = 0; i < n; i += n_last) {
        T max_num = a.data()[i];
        T sum = 0;

        for (std::size_t j = 0; j < n_last; ++j) {
            max_num = (std::max)(max_num, a.data()[i + j]);
        }
        for (std::size_t j = 0; j < n_last; ++j) {
            sum += std::exp(a.data()[i + j] - max_num);
        }
        for (std::size_t j = 0; j < n_last; ++j) {
            b.data()[i + j] = std::exp(a.data()[i + j] - max_num) / sum;
        }
    }

    return b;
}

// only see last dimension
template<typename T>
tensor<T> layernorm(const tensor<T>& x, const tensor<T>& weight, const tensor<T>& bias, T eps = T(1e-5)) {
    assert(x.is_contiguous() && "[layernorm] tensors must be contiguous");
    assert(weight.is_contiguous() && "[layernorm] tensors must be contiguous");
    assert(bias.is_contiguous() && "[layernorm] tensors must be contiguous");

    std::size_t n = x.total_size();
    std::size_t n_last = x.shape().back();

    assert(weight.shape().size() == 1 && weight.shape()[0] == n_last && "[layernorm] weight shape mismatch");
    assert(bias.shape().size() == 1 && bias.shape()[0] == n_last && "[layernorm] bias shape mismatch");

    tensor<T> y(x.shape());
    OMP_FOR
    for (std::size_t i = 0; i < n; i += n_last) {
        T mean = 0;
        T var = 0;
        for (std::size_t j = 0; j < n_last; ++j) {
            mean += x.data()[i + j];
        }
        mean /= T(n_last);
        for (std::size_t j = 0; j < n_last; ++j) {
            T tmp = x.data()[i + j] - mean;
            var += tmp * tmp;
        }
        var /= T(n_last);
        for (std::size_t j = 0; j < n_last; ++j) {
            y.data()[i + j] = (x.data()[i + j] - mean) / std::sqrt(var + eps) * weight.data()[j] + bias.data()[j];
        }
    }

    return y;
}

}