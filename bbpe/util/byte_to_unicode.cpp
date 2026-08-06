#include "util/byte_to_unicode.hpp"

#include <iostream>

namespace bbpe::util {

converter::converter() {
    int n = 0;
    for (int i = 0; i < 256; ++i) {
        bool visible = (i >= 33 && i <= 126) ||
                       (i >= 161 && i <= 172) ||
                       (i >= 174 && i <= 255);
        if (visible) {
            table[i] = i;
            reverse_table[i] = static_cast<char>(i);
        } else {
            table[i] = 256 + (n++);
            reverse_table[256 + n - 1] = static_cast<char>(i);
        }
    }
}

std::string converter::codepoint_to_utf8(std::uint32_t cp) const {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        // 110xxxxx 10xxxxxx
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

std::string converter::encode(const std::string& text) const {
    std::string result;
    for (char c : text) {
        result += codepoint_to_utf8(table[static_cast<unsigned char>(c)]);  
    }
    return result;
}

std::string converter::decode(const std::string& text) const {
    std::string result;
    for (std::size_t i = 0; i < text.size(); ++i) {
        auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            result.push_back(reverse_table.at(c));
            continue;
        }
        std::uint32_t cp = 0;
        if ((text[i] & 0xF8) == 0xF0) {
            cp = (text[i] & 0x07)      << 18;
            cp |= (text[i + 1] & 0x3F) << 12;
            cp |= (text[i + 2] & 0x3F) <<  6;
            cp |= (text[i + 3] & 0x3F);
            i += 3;
        } else if ((text[i] & 0xF0) == 0xE0) {
            cp = (text[i] & 0x0F)      << 12;
            cp |= (text[i + 1] & 0x3F) <<  6;
            cp |= (text[i + 2] & 0x3F);
            i += 2;
        } else if ((text[i] & 0xE0) == 0xC0) {
            cp = (text[i] & 0x1F)      <<  6;
            cp |= (text[i + 1] & 0x3F);
            i += 1;
        }
        result += reverse_table.at(cp);
    }
    return result;
}

}