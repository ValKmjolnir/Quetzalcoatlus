#include "util/utf8.hpp"

#include <iomanip>

namespace bbpe::utf8 {
std::uint32_t utf8_hdchk(const char head) {
    // RFC-2279 but now we use RFC-3629 so nbytes is less than 4
    const auto c = static_cast<std::uint8_t>(head);
    if ((c >> 5) == 0x06) { // 110x xxxx (10xx xxxx)^1
        return 1;
    }
    if ((c >> 4) == 0x0e) { // 1110 xxxx (10xx xxxx)^2
        return 2;
    }
    if ((c >> 3) == 0x1e) { // 1111 0xxx (10xx xxxx)^3
        return 3;
    }
    return 0;
}

std::ostream& print(std::ostream& os, const std::string& str) {
    for (std::uint64_t i = 0; i < str.length(); ++i) {
        auto c = static_cast<std::uint8_t>(str[i]);
        if (std::isprint(c)) {
            os << c;
            continue;
        }
        auto nbytes = utf8_hdchk(c);
        if (nbytes == 0 || i + nbytes >= str.length()) {
            os << "<\\x" << std::hex
               << static_cast<std::int32_t>(c & 0xff)
               << ">" << std::dec;
            continue;
        }
        std::string s(1, str[i]);
        for (std::uint64_t j = 1; j <= nbytes; ++j) {
            s += str[i + j];
        }
        i += nbytes;
        os << s;
    }
    return os;
}

}