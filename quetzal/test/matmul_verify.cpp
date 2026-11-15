#include "tensor/linalg.hpp"
#include "tensor/tensor.hpp"
#include "util/testing.hpp"

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace quetzal;

namespace {

struct oracle_result {
    tensor::tensor<double> value;
    tensor::tensor<double> abs_sum;
    std::size_t K = 0;
};

tensor::tensor<double> make_ref(std::size_t rows, std::size_t cols) {
    return tensor::tensor<double>({rows, cols});
}

oracle_result oracle_matmul_2d(const tensor::tensor<float>& a,
                               const tensor::tensor<float>& b) {
    const std::size_t M = a.shape(0);
    const std::size_t K = a.shape(1);
    const std::size_t N = b.shape(1);

    oracle_result r{make_ref(M, N), make_ref(M, N), K};

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double acc = 0.0;
            double acc_abs = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                double p = static_cast<double>(a[i][k]) *
                           static_cast<double>(b[k][j]);
                acc += p;
                acc_abs += std::fabs(p);
            }
            r.value.data()[i * N + j] = acc;
            r.abs_sum.data()[i * N + j] = acc_abs;
        }
    }
    return r;
}

oracle_result oracle_matmul_batch(const tensor::tensor<float>& lhs,
                                  const tensor::tensor<float>& rhs) {
    const std::size_t B = lhs.shape(0);
    const std::size_t M = lhs.shape(1);
    const std::size_t K = lhs.shape(2);
    const std::size_t N = rhs.shape(2);

    oracle_result r{make_ref(B * M, N), make_ref(B * M, N), K};

    for (std::size_t bi = 0; bi < B; ++bi) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                double acc = 0.0;
                double acc_abs = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    double p = static_cast<double>(lhs[bi][i][k]) *
                               static_cast<double>(rhs[bi][k][j]);
                    acc += p;
                    acc_abs += std::fabs(p);
                }
                const std::size_t row = bi * M + i;
                r.value.data()[row * N + j] = acc;
                r.abs_sum.data()[row * N + j] = acc_abs;
            }
        }
    }
    return r;
}

struct diff_result {
    double max_abs = 0.0;
    double max_ratio = 0.0;
    std::size_t worst_abs = 0;
    std::size_t worst_ratio = 0;
};

diff_result compare(const tensor::tensor<float>& got, const oracle_result& ref) {
    const double relax = 8.0 * static_cast<double>(ref.K + 1) *
                         static_cast<double>(FLT_EPSILON);
    const std::size_t n = got.total_size() < ref.value.total_size()
                              ? got.total_size()
                              : ref.value.total_size();

    diff_result d;
    for (std::size_t i = 0; i < n; ++i) {
        const double err = std::fabs(static_cast<double>(got.data()[i]) -
                                     ref.value.data()[i]);
        if (!(err <= d.max_abs)) {
            d.max_abs = err;
            d.worst_abs = i;
        }

        const double bound = relax * ref.abs_sum.data()[i];
        double ratio = 0.0;
        if (bound > 0.0) {
            ratio = err / bound;
        } else if (err > 0.0) {
            ratio = std::numeric_limits<double>::infinity();
        }
        if (!(ratio <= d.max_ratio)) {
            d.max_ratio = ratio;
            d.worst_ratio = i;
        }
    }
    return d;
}

void* offset_alloc(std::size_t size) {
    char* base = static_cast<char*>(std::malloc(size + 32));
    return base + 32;
}

void offset_free(void* ptr) {
    std::free(static_cast<char*>(ptr) - 32);
}

void fill_random(tensor::tensor<float>& t, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < t.total_size(); ++i) {
        t.data()[i] = dist(gen);
    }
}

tensor::tensor<float> make_tensor(std::vector<std::size_t> shape, unsigned seed) {
    tensor::tensor<float> t(shape);
    fill_random(t, seed);
    return t;
}

tensor::tensor<float> make_unaligned(std::vector<std::size_t> shape, unsigned seed) {
    tensor::tensor<float> t(shape, offset_alloc, offset_free);
    fill_random(t, seed);
    return t;
}

using matmul_fn = tensor::tensor<float> (*)(const tensor::tensor<float>&,
                                            const tensor::tensor<float>&);

bool shape_is_2d(const tensor::tensor<float>& t, std::size_t M, std::size_t N) {
    return t.ndim() == 2 && t.shape(0) == M && t.shape(1) == N;
}

bool shape_is_3d(const tensor::tensor<float>& t, std::size_t B, std::size_t M,
                 std::size_t N) {
    return t.ndim() == 3 && t.shape(0) == B && t.shape(1) == M && t.shape(2) == N;
}

void check_2d(const std::string& name, matmul_fn fn,
              const tensor::tensor<float>& a,
              const tensor::tensor<float>& b) {
    const tensor::tensor<float> got = fn(a, b);
    const diff_result d = compare(got, oracle_matmul_2d(a, b));

    std::cout << "[" << name << " M=" << a.shape(0) << " K=" << a.shape(1)
              << " N=" << b.shape(1) << "] max|diff| = " << std::scientific
              << std::setprecision(3) << d.max_abs
              << ", err/bound = " << d.max_ratio << std::endl;

    CHECK(shape_is_2d(got, a.shape(0), b.shape(1)));
    CHECK(d.max_ratio <= 1.0);
}

void check_batch(const std::string& name, matmul_fn fn,
                 const tensor::tensor<float>& lhs,
                 const tensor::tensor<float>& rhs) {
    const tensor::tensor<float> got = fn(lhs, rhs);
    const diff_result d = compare(got, oracle_matmul_batch(lhs, rhs));

    std::cout << "[" << name << " B=" << lhs.shape(0) << " M=" << lhs.shape(1)
              << " K=" << lhs.shape(2) << " N=" << rhs.shape(2)
              << "] max|diff| = " << std::scientific << std::setprecision(3)
              << d.max_abs << ", err/bound = " << d.max_ratio << std::endl;

    CHECK(shape_is_3d(got, lhs.shape(0), lhs.shape(1), rhs.shape(2)));
    CHECK(d.max_ratio <= 1.0);
}

void test_golden_2d(matmul_fn fn) {
    const tensor::tensor<float> a(std::vector<std::size_t>{2, 3},
                                  std::vector<float>{1, 2, 3, 4, 5, 6});
    const tensor::tensor<float> b(std::vector<std::size_t>{3, 4},
                                  std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8,
                                                     9, 10, 11, 12});
    const float expect[8] = {38, 44, 50, 56, 83, 98, 113, 128};

    const tensor::tensor<float> got = fn(a, b);
    const oracle_result ref = oracle_matmul_2d(a, b);
    for (int i = 0; i < 8; ++i) {
        CHECK(got.data()[i] == expect[i]);
        CHECK(ref.value.data()[i] == static_cast<double>(expect[i]));
    }
}

void test_golden_batch(matmul_fn fn) {
    const tensor::tensor<float> lhs(std::vector<std::size_t>{2, 1, 3},
                                    std::vector<float>{1, 2, 3, 1, 2, 3});
    const tensor::tensor<float> rhs(
        std::vector<std::size_t>{2, 3, 4},
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const float expect[8] = {38, 44, 50, 56, 38, 44, 50, 56};

    const tensor::tensor<float> got = fn(lhs, rhs);
    for (int i = 0; i < 8; ++i) {
        CHECK(got.data()[i] == expect[i]);
    }
}

struct case_2d {
    const char* name;
    std::size_t M;
    std::size_t K;
    std::size_t N;
};

const case_2d shape_cases_2d[] = {
    {"k1", 4, 1, 4},
    {"n1", 4, 5, 1},
    {"m1", 1, 1, 1},
    {"odd", 3, 5, 7},
    {"tail", 17, 33, 9},
    {"tail_n", 9, 64, 17},
    {"gemv_qkv", 1, 352, 1056},
    {"gemv_ffn_up", 1, 352, 1408},
    {"gemv_ffn_down", 1, 1408, 352},
    {"gemv_large_n", 1, 352, 8000},
    {"prefill_8", 8, 352, 1056},
    {"prefill_64", 64, 352, 352},
    {"prefill_150", 150, 352, 1056},
    {"m3_transition", 3, 352, 352},
};

void test_shapes_2d(matmul_fn fn) {
    for (const auto& c : shape_cases_2d) {
        check_2d(c.name, fn, make_tensor({c.M, c.K}, 1), make_tensor({c.K, c.N}, 2));
    }
}

void test_layouts(matmul_fn fn) {
    const tensor::tensor<float> a = make_tensor({8, 64}, 3);
    const tensor::tensor<float> bt = make_tensor({17, 64}, 4);
    CHECK(!bt.transpose(0, 1).is_contiguous());
    check_2d("b_transposed", fn, a, bt.transpose(0, 1));

    const tensor::tensor<float> at = make_tensor({64, 8}, 5);
    const tensor::tensor<float> b = make_tensor({64, 17}, 6);
    CHECK(!at.transpose(0, 1).is_contiguous());
    check_2d("a_transposed", fn, at.transpose(0, 1), b);

    check_2d("unaligned", fn, make_unaligned({8, 64}, 7),
             make_unaligned({64, 17}, 8));
}

struct case_batch {
    const char* name;
    std::size_t B;
    std::size_t M;
    std::size_t K;
    std::size_t N;
};

const case_batch shape_cases_batch[] = {
    {"qk_decode", 11, 1, 32, 20},
    {"attn_v_decode", 11, 20, 20, 32},
    {"qk_prefill", 11, 150, 32, 150},
};

void test_shapes_batch(matmul_fn fn) {
    for (const auto& c : shape_cases_batch) {
        check_batch(c.name, fn, make_tensor({c.B, c.M, c.K}, 11),
                    make_tensor({c.B, c.K, c.N}, 12));
    }

    const tensor::tensor<float> lhs = make_tensor({11, 1, 32}, 13);
    const tensor::tensor<float> kt = make_tensor({11, 20, 32}, 14);
    CHECK(!kt.transpose(1, 2).is_contiguous());
    check_batch("qk_decode_transposed", fn, lhs, kt.transpose(1, 2));
}

struct impl_entry {
    const char* name;
    matmul_fn fn;
};

const impl_entry impls_2d[] = {
    {"matmul_2d", &tensor::matmul_2d<float>},
    {"matmul_2d_tiling", &tensor::matmul_2d_tiling<float>},
};

const impl_entry impls_batch[] = {
    {"matmul_batch", &tensor::matmul_batch<float>},
};

} // namespace

int main() {
    for (const auto& impl : impls_2d) {
        std::cout << "=== " << impl.name << " ===" << std::endl;
        test_golden_2d(impl.fn);
        test_shapes_2d(impl.fn);
        test_layouts(impl.fn);
    }

    for (const auto& impl : impls_batch) {
        std::cout << "=== " << impl.name << " ===" << std::endl;
        test_golden_batch(impl.fn);
        test_shapes_batch(impl.fn);
    }

    return test::summary("matmul");
}
