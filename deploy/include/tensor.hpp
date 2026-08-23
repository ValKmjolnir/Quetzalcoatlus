#pragma once

#ifdef QGPT_USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include <type_traits>
#include <vector>

template<typename T>
class tensor {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");
private:
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    std::vector<T> data;
};
