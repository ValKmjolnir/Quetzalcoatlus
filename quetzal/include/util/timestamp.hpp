#pragma once

#include <chrono>
#include <ctime>
#include <string>

namespace quetzal::util {

class timestamp {
private:
    std::chrono::high_resolution_clock::time_point stamp_;

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
    std::chrono::nanoseconds elapsed_nano_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - stamp_);
    }
};

std::string format_time(std::time_t t);

}
