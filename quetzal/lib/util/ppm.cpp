#include "util/ppm.hpp"

#include <iostream>
#include <algorithm>

namespace quetzal::util {

static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

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
    float max_num = 0;
    for (std::size_t i = 0; i < x.total_size(); ++i) {
        max_num = (std::max)(max_num, std::pow(x.data()[i], 2.f));
    }

    for (std::size_t i = 0; i < width_; ++i) {
        write_pixel(0, 0, 0);
    }
    for (std::size_t i = 0; i < x.total_size(); ++i) {
        float t = (x.data()[i] < 0 ? -1 : 1) * std::pow(x.data()[i], 2.f) / max_num;
        t = std::clamp(t, -1.f, 1.f);
        float r = 0, g = 0, b = 0;
        if (x.data()[i] >= 0) {
            float u = t;
            r = lerp(0.02, 0.95, u);
            g = lerp(0.55, 0.95, u);
            b = lerp(0.85, 0.95, u);
        } else {
            float u = -t;
            r = lerp(0.95, 0.95, u);
            g = lerp(0.95, 0.75, u);
            b = lerp(0.95, 0.15, u);
        }
        write_pixel(r * 255, g * 255, b * 255);
        write_pixel(r * 255, g * 255, b * 255);
    }
}

}
