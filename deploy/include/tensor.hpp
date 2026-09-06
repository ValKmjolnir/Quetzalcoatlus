#pragma once

#ifdef QGPT_USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <stdexcept>
#include <random>
#include <type_traits>
#include <vector>
#include <memory>

#include <cassert>

namespace quetzal::tensor {

template<typename T>
class view {
private:
    const std::vector<std::size_t>* shape_;
    const std::vector<std::size_t>* strides_;
    T* data_;
    std::size_t dim_;
    std::size_t offset_;

public:
    view(const std::vector<std::size_t>& shape,
         const std::vector<std::size_t>& strides,
         T* data,
         std::size_t dim,
         std::size_t offset):
        shape_(&shape), strides_(&strides),
        data_(data), dim_(dim), offset_(offset) {}

    view operator[](std::size_t i) const {
        assert(dim_ < shape_->size());

        std::size_t offset = offset_ + (*strides_)[dim_] * i;
        return view<T>(*shape_, *strides_, data_, dim_ + 1, offset);
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
                out << ", ";
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

using alloc_func = void* (*)(std::size_t);
using free_func = void (*)(void*);

void *default_allocator(std::size_t size);
void default_deallocator(void *ptr);

template<typename T>
class tensor {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<void> data_;

private:
    std::size_t total_size() const {
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
    tensor(const std::vector<std::size_t>& shape,
           alloc_func allocator = default_allocator,
           free_func deallocator = default_deallocator): shape_(shape) {
        std::size_t s = total_size() * sizeof(T);
        data_ = std::shared_ptr<void>(allocator(s), deallocator);
        compute_strides();
    }

    tensor(std::initializer_list<std::size_t> shape,
           alloc_func allocator = default_allocator,
           free_func deallocator = default_deallocator): shape_(shape) {
        std::size_t n = total_size() * sizeof(T);
        data_ = std::shared_ptr<void>(allocator(n), deallocator);
        compute_strides();
    }

    tensor(const tensor& other) = default;

    view<T> operator[](std::size_t i) {
        assert(!shape_.empty());

        std::size_t offset = strides_[0] * i;
        return view<T>(shape_, strides_, data(), 1, offset);
    }

    void debug_init() {
        std::size_t n = total_size();
        for (std::size_t i = 0; i < n; i++) {
            data()[i] = i;
        }
    }

    T* data() { return static_cast<T*>(data_.get()); }

    const T* data() const { return static_cast<const T*>(data_.get()); }

    tensor<T> transpose(std::size_t i, std::size_t j) const {
        if (i == j) {
            return *this;
        }

        if (i >= shape_.size() || j >= shape_.size()) {
            throw std::out_of_range("transpose: dimension out of range");
        }

        tensor<T> ret = *this;
        std::swap(ret.shape_[i], ret.shape_[j]);
        std::swap(ret.strides_[i], ret.strides_[j]);
        return ret;
    }

    bool is_contiguous() const {
        std::size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
            if (strides_[i] != stride) {
                return false;
            }
            stride *= shape_[i];
        }
        return true;
    }

    tensor<T> contiguous() const {
        if (is_contiguous()) {
            return *this;
        }

        tensor<T> ret = tensor<T>(shape_);
        const std::size_t total = total_size();

        std::vector<std::size_t> idx;
        idx.resize(shape_.size());

        for (std::size_t i = 0; i < total; i++) {
            std::size_t rem = i;
            for (std::size_t j = 0; j < shape_.size(); j++) {
                idx[j] = rem / ret.strides_[j];
                rem %= ret.strides_[j];
            }
            std::size_t offset = 0;
            for (std::size_t j = 0; j < shape_.size(); j++) {
                offset += idx[j] * strides_[j];
            }
            ret.data()[i] = data()[offset];
        }

        return ret;
    }

public:
    void dump_info(std::ostream& out) {
        out << "shape: [ ";
        for (auto& i : shape_) {
            out << i << " ";
        }
        out << "], ";

        out << "strides: [ ";
        for (auto& i : strides_) {
            out << i << " ";
        }
        out << "]" << std::endl;
    }

    void dump(std::ostream& out, size_t indent = 0) {
        if (shape_.size() == 1) {
            out << "[";
            for (std::size_t i = 0; i < shape_[0]; i++) {
                out << data()[i];
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

}
