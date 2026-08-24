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
class tensor_view {
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    const T* data_;

public:
    tensor_view(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides, const T* data) {
        shape_ = shape;
        strides_ = strides;
        data_ = data;
    }

    tensor_view operator[](std::size_t i) const {
        if (i >= shape_[0]) {
            throw std::out_of_range("index out of range");
        }
        const T* offset = data_ + strides_[0] * i;
        std::vector<std::size_t> shape = std::vector<std::size_t>(shape_.begin() + 1, shape_.end());
        std::vector<std::size_t> strides = std::vector<std::size_t>(strides_.begin() + 1, strides_.end());
        return tensor_view<T>(shape, strides, offset);
    }

    operator T&() {
        return data_[0];
    }

    operator const T&() const {
        return data_[0];
    }

    void dump(std::ostream& out, size_t indent = 0) const {
        if (shape_.size() == 1) {
            for (std::size_t i = 0; i < indent; i++) {
                out << " ";
            }
            out << "[";
            for (std::size_t i = 0; i < shape_[0]; i++) {
                out << data_[i];
                if (i != shape_[0] - 1) {
                    out << ", ";
                }
            }
            out << "]";
        } else {
            for (std::size_t i = 0; i < indent; i++) {
                out << " ";
            }
            out << "[\n";
            for (std::size_t i = 0; i < shape_[0]; i++) {
                (*this)[i].dump(out, indent + 1);
                if (i != shape_[0] - 1) {
                    out << ",";
                }
                out << "\n";
            }
            for (std::size_t i = 0; i < indent; i++) {
                out << " ";
            }
            out << "]";
        }
    }
};

template<typename T>
class tensor {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<T> data_;

public:
    tensor(const std::vector<std::size_t>& shape) {
        shape_ = shape;
        strides_ = std::vector<std::size_t>(shape.size());
        strides_[shape.size() - 1] = 1;
        for (std::size_t i = shape.size() - 2; ; i--) {
            strides_[i] = strides_[i + 1] * shape[i + 1];
            if (i == 0) {
                break;
            }
        }

        data_ = std::vector<T>(strides_[0] * shape[0]);
    }

    tensor_view<T> operator[](std::size_t i) const {
        if (i >= shape_[0]) {
            throw std::out_of_range("index out of range");
        }
        const T* offset = data_.data() + strides_[0] * i;
        std::vector<std::size_t> shape = std::vector<std::size_t>(shape_.begin() + 1, shape_.end());
        std::vector<std::size_t> strides = std::vector<std::size_t>(strides_.begin() + 1, strides_.end());
        return tensor_view<T>(shape, strides, offset);
    }

    std::vector<T>& data() {
        return data_;
    }

    tensor& transpose(std::size_t i, std::size_t j) {
        std::swap(shape_[i], shape_[j]);
        std::swap(strides_[i], strides_[j]);
        return *this;
    }

    void dump(std::ostream& out, size_t indent = 0) const {
        out << "shape: ";
        for (auto& i : shape_) {
            out << i << " ";
        }
        out << std::endl;

        out << "strides: ";
        for (auto& i : strides_) {
            out << i << " ";
        }
        out << std::endl;

        for (std::size_t i = 0; i < indent; i++) {
            out << " ";
        }
        out << "[\n";
        for (std::size_t i = 0; i < shape_[0]; i++) {
            (*this)[i].dump(out, indent + 1);
            if (i != shape_[0] - 1) {
                out << ",";
            }
            out << "\n";
        }
        for (std::size_t i = 0; i < indent; i++) {
            out << " ";
        }
        out << "]" << std::endl;
    }
};
