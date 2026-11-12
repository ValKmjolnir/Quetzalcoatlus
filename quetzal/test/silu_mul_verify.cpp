#include "tensor/fusion.hpp"
#include "util/testing.hpp"

#include <iostream>
#include <cmath>

bool check_max_diff(const quetzal::tensor::tensor<float>& a,
                    const quetzal::tensor::tensor<float>& b,
                    const std::vector<std::size_t>& shape) {
    float max_diff = 0.0f;
    std::size_t arg = 0;
    for (std::size_t i = 0; i < a.total_size(); ++i) {
        float diff = std::fabs(a.data()[i] - b.data()[i]);
        if (diff > max_diff) {
            max_diff = diff;
            arg = i;
        }
    }

    std::cout << "[silu_mul_fusion shape=";
    for (std::size_t d = 0; d < shape.size(); ++d) {
        std::cout << (d == 0 ? "" : "x") << shape[d];
    }
    std::cout << "] max|diff| = " << max_diff << " @ " << arg << std::endl;

    if (max_diff != 0.0f) {
        std::cout << "[silu_mul_fusion] FAIL" << std::endl;
        return false;
    }
    return true;
}

bool test_silu_mul_fusion() {
    const std::vector<std::vector<std::size_t>> shapes = {
        {16},
        {4, 64},
        {2, 3, 8},
        {2, 3, 4, 8},
        {2, 3, 4, 8, 16},
    };
    for (const auto& shape : shapes) {
        quetzal::tensor::tensor<float> a(shape);
        quetzal::tensor::tensor<float> b(shape);
        quetzal::tensor::tensor<float> c(shape);
        quetzal::test::debug_init(a, 10.0f);
        quetzal::test::debug_init(b, 7.0f);
        quetzal::test::debug_init(c, 10.0f);

        quetzal::tensor::tensor<float> fused = quetzal::tensor::silu_mul_fusion(a, b);
        quetzal::tensor::tensor<float> ref   = quetzal::tensor::silu(a) * b;
        quetzal::tensor::inplace_silu_mul_fusion(c, b);

        if (!check_max_diff(fused, ref, shape)) {
            return false;
        }
        if (!check_max_diff(c, ref, shape)) {
            return false;
        }
    }
    std::cout << "[silu_mul_fusion] PASS (bit-identical to silu(a) * b)" << std::endl;
    return true;
}

int main() {
    CHECK(test_silu_mul_fusion());
    return quetzal::test::summary("silu_mul_fusion");
}