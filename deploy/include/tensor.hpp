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

#include <cassert>

template<typename T>
class tensor_view {
private:
    const std::vector<std::size_t>* shape_;
    const std::vector<std::size_t>* strides_;
    T* data_;
    std::size_t dim_;
    std::size_t offset_;

public:
    tensor_view(const std::vector<std::size_t>& shape,
                const std::vector<std::size_t>& strides,
                T* data,
                std::size_t dim,
                std::size_t offset):
                shape_(&shape), strides_(&strides),
                data_(data), dim_(dim), offset_(offset) {}

    tensor_view operator[](std::size_t i) const {
        assert(dim_ < shape_->size());

        std::size_t offset = offset_ + (*strides_)[dim_] * i;
        return tensor_view<T>(*shape_, *strides_, data_, dim_ + 1, offset);
    }

    operator T&() {
        assert(dim_ == shape_->size());
        return data_[offset_];
    }

    operator const T&() const {
        assert(dim_ == shape_->size());
        return data_[offset_];
    }

    void dump(std::ostream& out, size_t indent = 0) {
        if (dim_ == shape_->size()) {
            out << data_[offset_];
            return;
        }

        for (std::size_t i = 0; i < indent; i++) {
            out << " ";
        }
        out << "[";
        if (shape_->size() > 1 && dim_ < shape_->size() - 1) {
            out << "\n";
        }
        for (std::size_t i = 0; i < (*shape_)[dim_]; i++) {
            (*this)[i].dump(out, indent + 1);
            if (i != (*shape_)[dim_] - 1) {
                out << ",";
            }
            if (shape_->size() > 1 && dim_ < shape_->size() - 1) {
                out << "\n";
            }
        }
        if (shape_->size() > 1 && dim_ < shape_->size() - 1) {
            for (std::size_t i = 0; i < indent; i++) {
                out << " ";
            }
        }
        out << "]";
    }
};

template<typename T>
class tensor {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<T> data_;

private:
    std::size_t total_size() {
        std::size_t n = 1;
        for (auto& i : shape_) {
            n *= i;
        }
        return n;
    }

    void compute_strides() {
        strides_.resize(shape_.size());
        std::size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

public:
    tensor(const std::vector<std::size_t>& shape):
        shape_(shape), data_(total_size()) {
        compute_strides();
    }

    tensor(std::initializer_list<std::size_t> shape):
        shape_(shape), data_(total_size()) {
        compute_strides();
    }

    tensor_view<T> operator[](std::size_t i) {
        assert(!shape_.empty());

        std::size_t offset = strides_[0] * i;
        return tensor_view<T>(shape_, strides_, data_.data(), 1, offset);
    }

    std::vector<T>& vec() { return data_; }
    const std::vector<T>& vec() const { return data_; }

    T* data() { return data_.data(); }

    const T* data() const { return data_.data(); }

    tensor& transpose(std::size_t i, std::size_t j) {
        std::swap(shape_[i], shape_[j]);
        std::swap(strides_[i], strides_[j]);
        return *this;
    }

    void dump(std::ostream& out, size_t indent = 0) {
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

        if (shape_.size() == 1) {
            out << "[";
            for (std::size_t i = 0; i < shape_[0]; i++) {
                out << data_[i];
                if (i != shape_[0] - 1) {
                    out << ", ";
                }
            }
            out << "]" << std::endl;
            return;
        }

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
