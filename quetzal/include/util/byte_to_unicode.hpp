#pragma once

#include <array>
#include <string>
#include <cstdint>

#include "util/densemap.hpp"

namespace quetzal::util {

class converter {
private:
    std::array<std::uint32_t, 256> table;
    densemap<std::uint32_t, char> reverse_table;

private:
    std::string codepoint_to_utf8(std::uint32_t cp) const;

public:
    converter();
    std::string encode(const std::string& src) const;
    std::string decode(const std::string& src) const;
};

}