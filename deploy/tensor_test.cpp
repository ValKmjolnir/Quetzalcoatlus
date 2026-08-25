#include <iostream>
#include <vector>
#include "include/tensor.hpp"

void test_vector() {
    std::vector<std::size_t> shape = {2};
    tensor<float> a(shape);
    a.vec() = {1, 2};
    a.dump(std::cout);

    std::cout << "[" << a[0] << ", " << a[1] << "]" << std::endl;
}

void test() {
    tensor<float> a({2, 3, 4});
    a.vec() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    try {
        a.dump(std::cout);
        a.transpose(0, 1);
        a.dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    tensor<float> b({2, 3, 4});
    b.vec() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    try {
        b.dump(std::cout);
        b.transpose(0, 2);
        b.dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

int main() {
    test_vector();
    test();
    return 0;
}