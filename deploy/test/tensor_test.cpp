#include <iostream>
#include <sstream>
#include <vector>
#include "tensor.hpp"
#include "linalg.hpp"
#include "utils.hpp"

bool test_contiguous() {
    std::cout << "================== contiguous ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a);
    if (a.transpose(0, 1).is_contiguous()) {
        std::cout << "[contiguous] FAIL [a.transpose(0, 1).is_contiguous()]" << std::endl;
        return false;
    }
    std::cout << "[contiguous] PASS [a.transpose(0, 1).is_contiguous()]" << std::endl;
    
    if (!a.transpose(0, 1).transpose(0, 1).is_contiguous()) {
        std::cout << "[contiguous] FAIL [a.transpose(0, 1).transpose(0, 1).is_contiguous()]" << std::endl;
        return false;
    }
    std::cout << "[contiguous] PASS [a.transpose(0, 1).transpose(0, 1).is_contiguous()]" << std::endl;

    a.transpose(0, 1).dump_info(std::cout);
    a.transpose(0, 1).dump(std::cout);
    a.transpose(0, 1).contiguous().dump_info(std::cout);
    a.transpose(0, 1).contiguous().dump(std::cout);

    a.transpose(0, 2).dump_info(std::cout);
    a.transpose(0, 2).dump(std::cout);
    a.transpose(0, 2).contiguous().dump_info(std::cout);
    a.transpose(0, 2).contiguous().dump(std::cout);

    a.transpose(1, 2).dump_info(std::cout);
    a.transpose(1, 2).dump(std::cout);
    a.transpose(1, 2).contiguous().dump_info(std::cout);
    a.transpose(1, 2).contiguous().dump(std::cout);
    return true;
}

bool test_dump() {
    std::cout << "==================== dump =====================" << std::endl;
    std::vector<std::size_t> shape = {2};
    quetzal::tensor::tensor<float> a(shape);
    quetzal::utils::debug_init(a);

    std::stringstream ss1;
    a.dump(ss1);

    std::stringstream ss2;
    ss2 << "[" << a[0] << ", " << a[1] << "]" << std::endl;

    if (ss1.str() != ss2.str()) {
        std::cout << "[dump] FAIL [ss1.str() != ss2.str()]" << std::endl;
        return false;
    }
    std::cout << "[dump] PASS [ss1.str() == ss2.str()]" << std::endl;
    return true;
}

void test_transpose() {
    std::cout << "================== transpose ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a);
    try {
        a.dump_info(std::cout);
        a.dump(std::cout);
        a.transpose(0, 1).dump_info(std::cout);
        a.transpose(0, 1).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::utils::debug_init(b);
    try {
        b.dump_info(std::cout);
        b.dump(std::cout);
        b.transpose(0, 2).dump_info(std::cout);
        b.transpose(0, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> c({2, 3, 4});
    quetzal::utils::debug_init(c);
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
    std::cout << "===================== add =====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::utils::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = quetzal::tensor::add(a, b);
    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_mul() {
    std::cout << "===================== mul =====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::utils::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = quetzal::tensor::mul(a, b);
    c.dump_info(std::cout);
    c.dump(std::cout);

    quetzal::tensor::tensor<float> d = quetzal::tensor::mul(c, 100.0f);
    d.dump_info(std::cout);
    d.dump(std::cout);
}

void test_silu() {
    std::cout << "===================== silu ====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::silu(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_sigmoid() {
    std::cout << "=================== sigmoid ===================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::sigmoid(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_softmax() {
    std::cout << "=================== softmax ===================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 40});
    quetzal::utils::debug_init(a, 240.0f);
    a.dump_info(std::cout);
    a.dump(std::cout);
    quetzal::tensor::tensor<float> b = quetzal::tensor::softmax(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_layer_norm() {
    std::cout << "================== layernorm ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 40});
    quetzal::utils::debug_init(a, 240.0f);
    quetzal::tensor::tensor<float> weights({40});
    quetzal::utils::debug_init(weights, 80.0f);
    quetzal::tensor::tensor<float> bias({40});
    quetzal::utils::debug_init(bias, 80.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::layernorm(a, weights, bias, 1e-5f);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_2d_matmul() {
    std::cout << "=================== 2d matmul ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3});
    quetzal::utils::debug_init(a);
    quetzal::tensor::tensor<float> b({3, 4});
    quetzal::utils::debug_init(b);
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
    return 0;
}