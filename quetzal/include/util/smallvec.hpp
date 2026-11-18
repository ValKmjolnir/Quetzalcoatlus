#pragma once

#include <stdexcept>
#include <array>
#include <cstring>
#include <vector>
#include <initializer_list>

namespace quetzal::util {

template<typename T>
class smallvec {
private:
    std::array<T, 8> data_;
    size_t size_ = 0;

public:
    bool operator==(const smallvec<T>& s) const {
        if (size_ != s.size_) {
            return false;
        }
        return std::memcmp(data_.data(), s.data_.data(), size_ * sizeof(T)) == 0;
    }
    T& operator[](size_t i) {
        return data_[i];
    }
    const T& operator[](size_t i) const {
        return data_[i];
    }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    T& at(size_t i) { return data_[i]; }
    const T& at(size_t i) const { return data_[i]; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }
    void resize(size_t n) {
        if (n > 8) {
            throw std::out_of_range("smallvec: vector too large");
        }
        size_ = n;
    }
    T* begin() { return data_.begin(); }
    T* end() { return data_.begin() + size_; }
    const T* begin() const { return data_.begin(); }
    const T* end() const { return data_.begin() + size_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

public:
    smallvec() = default;
    smallvec(const smallvec<T>& s): size_(s.size_) {
        std::memcpy(data_.data(), s.data_.data(), s.size_ * sizeof(T));
    }
    smallvec(const std::vector<T>& v): size_(v.size()) {
        if (size_ > 8) {
            throw std::out_of_range("smallvec: vector too large");
        }
        std::memcpy(data_.data(), v.data(), v.size() * sizeof(T));
    }
    smallvec(std::initializer_list<T>& il): size_(il.size()) {
        if (size_ > 8) {
            throw std::out_of_range("smallvec: vector too large");
        }
        std::memcpy(data_.data(), il.begin(), il.size() * sizeof(T));
    }
};

}
