#pragma once

#include <cstdint>
#include <string>
#include <iostream>

namespace bbpe::utf8 {

std::uint32_t utf8_hdchk(const char head);

std::ostream& print(std::ostream& os, const std::string&);

}