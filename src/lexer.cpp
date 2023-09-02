#include "lexer.h"

#include <iostream>

namespace quetza {

constexpr bool lexer::isquote(const char c) {
    return c == '\"';
}

constexpr bool lexer::isidentifier(const char c) {
    return std::isalpha(c) || c == '_';
}

constexpr bool lexer::isnumber(const char c) {
    return std::isdigit(c);
}

void lexer::readfile(const std::string& filename) {
    std::stringstream ss;
    std::ifstream in(filename, std::ios::binary);
    if (in.fail()) {
        std::cerr << "cannot open file <" << filename << ">\n";
        return;
    }
    ss << in.rdbuf();
    source = ss.str();
}

void lexer::scan(const std::string& filename) {
    readfile(filename);
    if (source.empty()) {
        return;
    }

    size_t index = 0;
    while(index < source.length()) {
        if (isnumber(source[index])) {
            ++index;
        } else if (isidentifier(source[index])) {
            ++index;
        } else if (isquote(source[index])) {
            ++index;
        } else {
            std::cerr << "invalid character ";
            if (std::isgraph(source[index])) {
                std::cerr << source[index] << "\n";
            } else {
                std::cerr << "0x" << std::hex;
                std::cerr << (uint32_t)source[index] << std::dec << "\n";
            }
            ++index;
        }
    }

    tokens.push_back({tok_type::eof, ""});
}

}
