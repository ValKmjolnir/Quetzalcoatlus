#include <iostream>
#include <vector>
#include "include/tensor.hpp"

bool test_contiguous() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    a.debug_init();
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

    return true;
}

void test_vector() {
    std::vector<std::size_t> shape = {2};
    quetzal::tensor::tensor<float> a(shape);
    a.debug_init();
    a.dump(std::cout);

    std::cout << "[" << a[0] << ", " << a[1] << "]" << std::endl;
}

void test() {
    quetzal::tensor::tensor<float> a({2, 3, 4});
    a.debug_init();
    try {
        a.dump(std::cout);
        a.transpose(0, 1).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> b({2, 3, 4});
    b.debug_init();
    try {
        b.dump(std::cout);
        b.transpose(0, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> c({2, 3, 4});
    c.debug_init();
    try {
        c.dump(std::cout);
        c.transpose(1, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

int main() {
    test_vector();
    test();
    test_contiguous();
    return 0;
}