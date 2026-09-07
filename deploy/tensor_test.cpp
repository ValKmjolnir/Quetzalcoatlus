#include <iostream>
#include <vector>
#include "include/tensor.hpp"
#include "include/linalg.hpp"
#include "include/utils.hpp"

bool test_contiguous() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
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

void test_vector() {
    std::vector<std::size_t> shape = {2};
    quetzal::tensor::tensor<float> a(shape);
    quetzal::utils::debug_init(a, 10.0f);
    a.dump(std::cout);

    std::cout << "[" << a[0] << ", " << a[1] << "]" << std::endl;
}

void test() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    try {
        a.dump_info(std::cout);
        a.dump(std::cout);
        a.transpose(0, 1).dump_info(std::cout);
        a.transpose(0, 1).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::utils::debug_init(b, 10.0f);
    try {
        b.dump_info(std::cout);
        b.dump(std::cout);
        b.transpose(0, 2).dump_info(std::cout);
        b.transpose(0, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> c({2, 3, 4});
    quetzal::utils::debug_init(c, 10.0f);
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
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::utils::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = quetzal::tensor::add(a, b);
    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_mul() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::utils::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = quetzal::tensor::mul(a, b);
    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_silu() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::silu(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_sigmoid() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::utils::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::sigmoid(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

int main() {
    test_vector();
    test();
    test_contiguous();

    test_add();
    test_mul();
    test_silu();
    test_sigmoid();
    return 0;
}