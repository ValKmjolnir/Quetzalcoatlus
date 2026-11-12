#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include "tensor/tensor.hpp"
#include "tensor/linalg.hpp"
#include "tensor/fusion.hpp"
#include "tensor/rope.hpp"
#include "util/testing.hpp"

void test_contiguous() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a);
    CHECK(!a.transpose(0, 1).is_contiguous());
    CHECK(a.transpose(0, 1).transpose(0, 1).is_contiguous());

    auto do_test = [](quetzal::tensor::tensor<float> &x,
                      std::size_t axis0,
                      std::size_t axis1) -> bool {
        auto lhs = x.transpose(axis0, axis1);
        auto rhs = x.transpose(axis0, axis1).contiguous();

        std::stringstream linfo;
        lhs.dump_info(linfo);
        std::stringstream lss;
        lhs.dump(lss);
        std::stringstream rinfo;
        rhs.dump_info(rinfo);
        std::stringstream rss;
        rhs.dump(rss);

        return lss.str() == rss.str() && linfo.str() != rinfo.str();
    };

    CHECK(do_test(a, 0, 1));
    CHECK(do_test(a, 0, 2));
    CHECK(do_test(a, 1, 2));
}

void test_dump() {
    std::vector<std::size_t> shape = {4};
    quetzal::tensor::tensor<float> a(shape);
    quetzal::test::debug_init(a);

    std::stringstream ss1;
    a.dump(ss1);

    std::stringstream ss2;
    ss2 << "[" << a[0] << ", " << a[1] << ", "
        << a[2] << ", " << a[3] << "]\n";

    CHECK(ss1.str() == ss2.str());
}

void test_transpose() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a);
    try {
        a.dump_info(std::cout);
        a.dump(std::cout);
        a.transpose(0, 1).dump_info(std::cout);
        a.transpose(0, 1).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::test::debug_init(b);
    try {
        b.dump_info(std::cout);
        b.dump(std::cout);
        b.transpose(0, 2).dump_info(std::cout);
        b.transpose(0, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> c({2, 3, 4});
    quetzal::test::debug_init(c);
    try {
        c.dump_info(std::cout);
        c.dump(std::cout);
        c.transpose(1, 2).dump_info(std::cout);
        c.transpose(1, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

void test_add() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::test::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = a + b;
    
    quetzal::tensor::tensor<float> d({2, 3, 4});
    quetzal::test::debug_init(d, 5.0f);

    for (std::size_t i = 0; i < c.total_size(); ++i) {
        CHECK_NEAR(c.data()[i], a.data()[i] + b.data()[i], 0.f);
        CHECK_NEAR(c.data()[i], d.data()[i], 0.f);
    }
}

void test_mul() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::test::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = a * b;
    c.dump_info(std::cout);
    c.dump(std::cout);

    quetzal::tensor::tensor<float> d = c * 100.0f;
    d.dump_info(std::cout);
    d.dump(std::cout);
}

void test_silu() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::silu(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_sigmoid() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::sigmoid(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_softmax() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3, 40});
    quetzal::test::debug_init(a, 240.0f);
    a.dump_info(std::cout);
    a.dump(std::cout);
    quetzal::tensor::tensor<float> b = quetzal::tensor::softmax(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_layer_norm() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3, 40});
    quetzal::test::debug_init(a, 240.0f);
    quetzal::tensor::tensor<float> weights({40});
    quetzal::test::debug_init(weights, 80.0f);
    quetzal::tensor::tensor<float> bias({40});
    quetzal::test::debug_init(bias, 80.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::layernorm(a, weights, bias, 1e-5f);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_2d_matmul() {
    // TODO: test
    quetzal::tensor::tensor<float> a({2, 3});
    quetzal::test::debug_init(a);
    quetzal::tensor::tensor<float> b({3, 4});
    quetzal::test::debug_init(b);
    quetzal::tensor::tensor<float> c = quetzal::tensor::matmul_2d(a, b);

    a.dump(std::cout);
    b.dump(std::cout);
    c.dump_info(std::cout);
    c.dump(std::cout);

    quetzal::tensor::tensor<float> d = b.transpose(0, 1).contiguous();
    quetzal::tensor::tensor<float> e = quetzal::tensor::matmul_2d(a, d.transpose(0, 1));

    e.dump_info(std::cout);
    e.dump(std::cout);
}

void test_2d_matmul_perf() {
    quetzal::tensor::tensor<float> a({352, 3520 * 4});
    quetzal::test::debug_init(a);
    quetzal::tensor::tensor<float> b({3520 * 4, 352});
    quetzal::test::debug_init(b);

    std::cout << "=============== 2d matmul (perf) ==============" << std::endl;
    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
    quetzal::tensor::tensor<float> c = quetzal::tensor::matmul_2d(a, b);
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "matmul_2d time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms" << std::endl;

    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_matmul_batch() {
    std::cout << "================= matmul batch ================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::tensor<float> b({2, 4, 2});
    quetzal::test::debug_init(a);
    quetzal::test::debug_init(b);

    quetzal::tensor::tensor<float> c = quetzal::tensor::matmul_batch(a, b);
    a.dump(std::cout);
    b.dump(std::cout);
    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_causal_mask() {
    std::cout << "=================== casual mask ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 5, 5});
    quetzal::test::debug_init(a, 50.0f);
    quetzal::tensor::apply_causal_mask(a);

    a.dump_info(std::cout);
    a.dump(std::cout);
}

void test_reshape() {
    std::cout << "==================== reshape ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::test::debug_init(a);
    quetzal::tensor::tensor<float> b = a.reshape({2, 12});

    a.dump_info(std::cout);
    a.dump(std::cout);
    b.dump_info(std::cout);
    b.dump(std::cout);

    try {
        auto tmp = a.transpose(1, 2).reshape({2, 12});
    } catch (std::exception &e) {
        std::cout << "expected exception:\n";
        std::cout << "  - " << e.what() << std::endl;
    }
}

bool test_embedding_gather() {
    std::cout << "=============== embedding gather ==============" << std::endl;
    quetzal::tensor::tensor<float> embedding({8, 352});
    quetzal::test::debug_init(embedding);
    std::vector<std::uint32_t> indices({0, 2, 4, 6});

    quetzal::tensor::tensor<float> got = quetzal::tensor::embedding_gather(embedding, indices);
    std::vector<float> expected;
    for (auto i : indices) {
        for (std::size_t j = 0; j < 352; ++j) {
            expected.push_back(i * 352.0f + j);
        }
    }

    bool ok = true;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (got.data()[i] != expected[i]) {
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "[embedding_gather] PASS" << std::endl;
    } else {
        std::cout << "[embedding_gather] FAIL" << std::endl;
    }
    return ok;
}

void test_rope() {
    std::cout << "===================== rope ====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 64, 44});
    quetzal::test::debug_init(a);
    
    quetzal::tensor::rope<float> rope(64, 44);
    rope.apply(a);

    a.dump_info(std::cout);
    a.dump(std::cout);
}

bool test_topk() {
    std::cout << "===================== topk ====================" << std::endl;
    quetzal::tensor::tensor<float> a({6});
    quetzal::test::debug_init(a);

    auto value = quetzal::tensor::topk(a, 5);
    if (value == 1.0f) {
        std::cout << "[topk] PASS" << std::endl;
    } else {
        std::cout << "[topk] FAIL" << std::endl;
    }
    return value == 1.0f;
}

bool test_multinomial() {
    std::cout << "================== multinomial ================" << std::endl;
    std::mt19937_64 rng(114514);

    quetzal::tensor::tensor<float> a({6});
    quetzal::test::zero(a);
    a.data()[0] = 1.0f;
    for (int i = 0; i < 1000; ++i) {
        if (quetzal::tensor::multinomial(a, rng) != 0) {
            std::cout << "[multinomial] FAIL [expect 0]" << std::endl;
            return false;
        }
    }

    quetzal::test::zero(a);
    a.data()[2] = 1.0f;
    for (int i = 0; i < 1000; ++i) {
        if (quetzal::tensor::multinomial(a, rng) != 2) {
            std::cout << "[multinomial] FAIL [expect 2]" << std::endl;
            return false;
        }
    }

    std::cout << "[multinomial] PASS" << std::endl;
    return true;
}

int main() {
    test_dump();
    test_transpose();
    test_contiguous();

    test_add();
    test_mul();
    test_silu();
    test_sigmoid();
    test_softmax();
    test_layer_norm();
    test_2d_matmul();
    test_2d_matmul_perf();

    test_causal_mask();
    test_reshape();
    CHECK(test_embedding_gather());
    test_rope();

    test_matmul_batch();

    CHECK(test_topk());
    CHECK(test_multinomial());
    return quetzal::test::summary("tensor_test");
}