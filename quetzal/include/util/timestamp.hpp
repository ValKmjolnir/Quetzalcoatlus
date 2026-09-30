#pragma once

#include <chrono>

namespace quetzal::util {

class timestamp {
private:
    std::chrono::steady_clock::time_point stamp_;

public:
    timestamp() : stamp_(std::chrono::high_resolution_clock::now()) {}
    void stamp() {
        stamp_ = std::chrono::high_resolution_clock::now();
    }
    std::chrono::microseconds elapsed_micro_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - stamp_);
    }
    std::chrono::milliseconds elapsed_milli_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - stamp_);
    }
};

}
