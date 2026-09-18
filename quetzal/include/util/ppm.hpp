#pragma once

#include "tensor/tensor.hpp"

#include <fstream>
#include <string>

namespace quetzal::util {

class ppm_writer {
private:
    std::ofstream out_;
    std::size_t width_;
    std::size_t height_;
    std::size_t total_bytes_;

private:
    void write_pixel(unsigned char r, unsigned char g, unsigned char b) {
        ++total_bytes_;
        out_ << r << g << b;
    }

public:
    ppm_writer(const std::string& path, std::size_t width, std::size_t height);
    ~ppm_writer();
    void write(const tensor::tensor<float>& x);
};

}
