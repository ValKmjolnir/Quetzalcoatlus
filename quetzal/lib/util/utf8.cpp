#include "util/utf8.hpp"

namespace quetzal::utf8 {

static std::string to_hex(std::uint8_t c) {
    std::string s;
    s += "0123456789abcdef"[(c >> 4) & 0x0f];
    s += "0123456789abcdef"[c & 0x0f];
    return s;
}

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
            os << "<\\x" << to_hex(c) << ">";
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

std::string utf8_stream_decoder::feed(const std::string& bytes) {
    std::string input = pending_ + bytes;
    pending_.clear();

    std::string output;
    std::size_t i = 0;
    while (i < input.length()) {
        auto c = static_cast<std::uint8_t>(input[i]);
        if ((c >> 7) == 0) {
            output += input[i];
            ++i;
            continue;
        }

        auto nbytes = utf8_hdchk(c);
        if (!nbytes) {
            output += "<\\x" + to_hex(c) + ">";
            ++i;
            continue;
        }
        if (i + nbytes >= input.length()) {
            pending_ = input.substr(i);
            break;
        }

        bool valid = true;
        for (std::uint32_t j = 1; j <= nbytes; ++j) {
            auto d = static_cast<std::uint8_t>(input[i + j]);
            if ((d >> 6) != 0x2) { // 10xx xxxx
                valid = false;
                break;
            }
        }
        if (valid) {
            output += input.substr(i, nbytes + 1);
        } else {
            output += "<\\x";
            for (std::uint32_t j = 0; j <= nbytes; ++j) {
                auto d = static_cast<std::uint8_t>(input[i + j]);
                output += to_hex(d);
            }
            output += ">";
        }
        i += nbytes + 1;
    }
    return output;
}

}