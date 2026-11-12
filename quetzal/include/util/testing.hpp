#pragma once

#include <cmath>
#include <iostream>
#include <string>

namespace quetzal::test {

template<typename T>
void debug_init(quetzal::tensor::tensor<T>& input, T div_base = T(1)) {
    std::size_t n = input.total_size();
    for (std::size_t i = 0; i < n; i++) {
        input.data()[i] = i / div_base;
    }
}

template<typename T>
void zero(quetzal::tensor::tensor<T>& input) {
    std::memset(input.data(), 0, input.total_size() * sizeof(T));
}

inline int qz_test_count = 0;
inline int qz_test_failed = 0;

inline void fail(const char* file, int line, std::string msg) {
    ++quetzal::test::qz_test_failed;
    std::cerr << "[FAIL] " << file << ":" << line << ": " << msg << std::endl;
}

inline int summary(const char* test_name) {
    std::cerr << (quetzal::test::qz_test_failed == 0 ? "[PASS]" : "[FAIL]")
              << " " << test_name << ": "
              << quetzal::test::qz_test_count << " tests, "
              << quetzal::test::qz_test_failed << " failures"
              << std::endl;
    return quetzal::test::qz_test_failed == 0 ? 0 : 1;
}

}

#define CHECK(cond) \
    do {                                                                 \
        ++quetzal::test::qz_test_count;                                  \
        if (!(cond)) { quetzal::test::fail(__FILE__, __LINE__, #cond); } \
    } while(0)

#define CHECK_NEAR(a, b, eps) \
    do {                                                     \
        ++quetzal::test::qz_test_count;                      \
        double a_ = (a), b_ = (b), eps_ = (eps);             \
        if (!(std::fabs(a_ - b_) <= eps_)) {                 \
            quetzal::test::fail(__FILE__, __LINE__,          \
                "|" #a " - " #b "| = " +                     \
                std::to_string(std::fabs(a_ - b_)) + " > " + \
                std::to_string(eps_)                         \
            );                                               \
        }                                                    \
    } while(0)

