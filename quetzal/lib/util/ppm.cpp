#include "util/ppm.hpp"

#include <iostream>
#include <algorithm>

namespace quetzal::util {

ppm_writer::ppm_writer(const std::string& path, std::size_t width, std::size_t height) :
    out_(path, std::ios::binary),
    width_(width), height_(height), total_bytes_(0) {
    if (!out_) {
        throw std::runtime_error("Could not open file " + path);
    }
    out_ << "P6\n" << width << " " << height << "\n255\n";
}

ppm_writer::~ppm_writer() {
    out_.close();
    std::cout << "[Info: ppm_writer] write " << total_bytes_ << " bytes"
              << " | " << width_ << "x" << height_
              << " | actual height=" << total_bytes_ / width_ << "\n";
}

void ppm_writer::write(const tensor::tensor<float>& x) {
    float max_num = -std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < x.total_size(); ++i) {
        max_num = (std::max)(max_num, std::abs(x.data()[i]));
    }

    for (std::size_t i = 0; i < width_; ++i) {
        write_pixel(0, 0, 0);
    }
    for (std::size_t i = 0; i < x.total_size(); ++i) {
        const float t = std::clamp(x.data()[i] / max_num, -1.f, 1.f);
        const float meg = std::sqrt(std::abs(t));

        if (t >= 0) {
            write_pixel(190.f * meg, 100.f * meg, 30.f * meg);
            write_pixel(190.f * meg, 100.f * meg, 30.f * meg);
        } else {
            write_pixel(30.f * meg, 100.f * meg, 190.f * meg);
            write_pixel(30.f * meg, 100.f * meg, 190.f * meg);
        }
    }
}

}
