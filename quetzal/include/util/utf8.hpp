#pragma once

#include <cstdint>
#include <string>
#include <iostream>

namespace quetzal::utf8 {

std::uint32_t utf8_hdchk(const char head);
std::uint32_t utf8_str_wcwidth(const std::string& str);
std::ostream& print(std::ostream& os, const std::string& str);

class utf8_stream_decoder {
private:
    std::string pending_ = "";

public:
    std::string feed(const std::string& bytes);
};

}
