#include "omp.hpp"
#include "tensor.hpp"
#include "assert.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

namespace quetzal::tensor {

template<typename T>
bool shape_equal(const tensor<T>& a, const tensor<T>& b) {
    return a.shape() == b.shape();
}

template<typename T>
tensor<T> add(const tensor<T>& a, const tensor<T>& b) {
    QUETZAL_ASSERT(shape_equal(a, b), "[add] shape mismatch");
    QUETZAL_ASSERT(a.is_contiguous() && b.is_contiguous(), "[add] tensors must be contiguous");

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
    QUETZAL_ASSERT(shape_equal(a, b), "[mul] shape mismatch");
    QUETZAL_ASSERT(a.is_contiguous() && b.is_contiguous(), "[mul] tensors must be contiguous");

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
    QUETZAL_ASSERT(a.is_contiguous(), "[mul] tensor must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> c(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        c.data()[i] = a.data()[i] * b;
    }

    return c;
}

template<typename T>
tensor<T> div(const tensor<T>& a, const tensor<T>& b) {
    QUETZAL_ASSERT(shape_equal(a, b), "[div] shape mismatch");
    QUETZAL_ASSERT(a.is_contiguous() && b.is_contiguous(), "[div] tensors must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> c(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        c.data()[i] = a.data()[i] / b.data()[i];
    }

    return c;
}

template<typename T>
tensor<T> div(const tensor<T>& a, T b) {
    QUETZAL_ASSERT(a.is_contiguous(), "[div] tensor must be contiguous");

    std::size_t n = a.total_size();
    tensor<T> c(a.shape());

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        c.data()[i] = a.data()[i] / b;
    }

    return c;
}

template<typename T>
tensor<T> silu(const tensor<T>& a) {
    QUETZAL_ASSERT(a.is_contiguous(), "[silu] tensors must be contiguous");

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
    QUETZAL_ASSERT(a.is_contiguous(), "[sigmoid] tensors must be contiguous");

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
    QUETZAL_ASSERT(a.is_contiguous(), "[softmax] tensors must be contiguous");

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
    QUETZAL_ASSERT(x.is_contiguous(), "[layernorm] tensors must be contiguous");
    QUETZAL_ASSERT(weight.is_contiguous(), "[layernorm] tensors must be contiguous");
    QUETZAL_ASSERT(bias.is_contiguous(), "[layernorm] tensors must be contiguous");

    std::size_t n = x.total_size();
    std::size_t n_last = x.shape().back();

    QUETZAL_ASSERT(weight.shape().size() == 1 && weight.shape()[0] == n_last, "[layernorm] weight shape mismatch");
    QUETZAL_ASSERT(bias.shape().size() == 1 && bias.shape()[0] == n_last, "[layernorm] bias shape mismatch");

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

template<typename T>
tensor<T> matmul_2d(const tensor<T>& a, const tensor<T>& b) {
    QUETZAL_ASSERT(a.shape().size() == 2, "[matmul_2d] a shape mismatch");
    QUETZAL_ASSERT(b.shape().size() == 2, "[matmul_2d] b shape mismatch");

    QUETZAL_ASSERT(a.shape()[1] == b.shape()[0], "[matmul_2d] shape mismatch");

    const std::size_t M = a.shape()[0];
    const std::size_t K = a.shape()[1];
    const std::size_t N = b.shape()[1];

    const std::size_t sa0 = a.strides()[0], sa1 = a.strides()[1];
    const std::size_t sb0 = b.strides()[0], sb1 = b.strides()[1];

    tensor<T> c(std::vector<std::size_t>{M, N});
    std::memset(c.data(), 0, c.total_size() * sizeof(T));

    OMP_FOR
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            T aik = a.data()[i * sa0 + k * sa1];
            for (std::size_t j = 0; j < N; ++j) {
                c.data()[i * N + j] += aik * b.data()[k * sb0 + j * sb1];
            }
        }
    }

    return c;
}

template<typename T>
tensor<T> matmul_batch(const tensor<T>& lhs, const tensor<T>& rhs) {
    QUETZAL_ASSERT(lhs.shape().size() == 3, "[matmul_batch] lhs shape mismatch");
    QUETZAL_ASSERT(rhs.shape().size() == 3, "[matmul_batch] rhs shape mismatch");
    const std::size_t B = lhs.shape()[0];
    const std::size_t M = lhs.shape()[1];
    const std::size_t K = lhs.shape()[2];
    const std::size_t N = rhs.shape()[2];
    QUETZAL_ASSERT(rhs.shape()[0] == B, "[matmul_batch] batch mismatch");
    QUETZAL_ASSERT(rhs.shape()[1] == K, "[matmul_batch] k mismatch");

    tensor<T> c(std::vector<std::size_t>{B, M, N});
    std::memset(c.data(), 0, c.total_size() * sizeof(T));

    const std::size_t sa0 = lhs.strides()[0], sa1 = lhs.strides()[1], sa2 = lhs.strides()[2];
    const std::size_t sb0 = rhs.strides()[0], sb1 = rhs.strides()[1], sb2 = rhs.strides()[2];

    OMP_FOR
    for (std::size_t bi = 0; bi < B * M; ++bi) {
        const std::size_t b = bi / M;
        const std::size_t i = bi % M;
        T* c_row = c.data() + b * M * N + i * N;
        for (std::size_t k = 0; k < K; ++k) {
            T aik = lhs.data()[b * sa0 + i * sa1 + k * sa2];
            for (std::size_t j = 0; j < N; ++j) {
                c_row[j] += aik * rhs.data()[b * sb0 + k * sb1 + j * sb2];
            }
        }
    }

    return c;
}

template<typename T>
void causal_mask(tensor<T>& x) {
    QUETZAL_ASSERT(x.shape().size() >= 2, "[causal_mask] shape mismatch");
    QUETZAL_ASSERT(x.is_contiguous(), "[causal_mask] tensors must be contiguous");

    const std::size_t last_dim = x.shape()[x.shape().size() - 1];
    const std::size_t second_last_dim = x.shape()[x.shape().size() - 2];
    QUETZAL_ASSERT(last_dim == second_last_dim, "[causal_mask] shape mismatch");

    const std::size_t n = x.total_size();
    const std::size_t N = last_dim;

    OMP_FOR
    for (std::size_t base = 0; base < n; base += N * N) {
        for (std::size_t i = 0; i < N * N; ++i) {
            x.data()[base + i] = (i / N) < (i % N)
                ? -std::numeric_limits<T>::infinity()
                : x.data()[base + i];
        }
    }
}

template<typename T>
tensor<T> embedding_gather(const tensor<T>& weight, const std::vector<std::size_t>& indices) {
    QUETZAL_ASSERT(weight.shape().size() == 2, "[embedding_gather] shape mismatch");
    QUETZAL_ASSERT(weight.is_contiguous(), "[embedding_gather] tensors must be contiguous");
    QUETZAL_ASSERT(!indices.empty(), "[embedding_gather] indices cannot be empty");

    const std::size_t n = indices.size();
    tensor<T> x(std::vector<std::size_t>{n, weight.shape()[1]});
    for (auto id : indices) {
        QUETZAL_ASSERT(id < weight.shape()[0], "[embedding_gather] index out of range");
    }

    OMP_FOR
    for (std::size_t i = 0; i < n; ++i) {
        std::memcpy(x.data() + i * weight.shape()[1],
                    weight.data() + indices[i] * weight.shape()[1],
                    weight.shape()[1] * sizeof(T));
    }

    return x;
}

template<typename T>
T topk(const quetzal::tensor::tensor<T>& logits, std::size_t k) {
    const std::size_t n = logits.total_size();
    QUETZAL_ASSERT(logits.shape().size() == 1, "[topk] logits shape mismatch");
    QUETZAL_ASSERT(k > 0 && k <= n, "[topk] k out of range");

    std::vector<T> buf(logits.data(), logits.data() + n);
    std::nth_element(buf.begin(), buf.begin() + (n - k), buf.end());
    return buf[n - k];
}

template<typename T>
std::size_t multinomial(const tensor<T>& probs, std::mt19937_64& rng) {
    QUETZAL_ASSERT(probs.shape().size() == 1, "[multinomial] probs shape mismatch");
    QUETZAL_ASSERT(probs.is_contiguous(), "[multinomial] tensors must be contiguous");

    const std::size_t n = probs.total_size();
    T total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total += probs.data()[i];
    }
    QUETZAL_ASSERT(total > 0, "[multinomial] total probability must be positive");

    std::uniform_real_distribution<T> dist(T(0), total);
    const T r = dist(rng);

    T sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += probs.data()[i];
        if (sum > r) {
            return i;
        }
    }

    return n - 1;
}

}