#include "lexer.h"

#include <iostream>
#include <unordered_map>

namespace quetza {

constexpr bool lexer::is_quote(const char c) {
    return c == '\"';
}

constexpr bool lexer::is_identifier(const char c) {
    return std::isalpha(c) || c == '_';
}

constexpr bool lexer::is_number(const char c) {
    return std::isdigit(c);
}

constexpr bool lexer::is_single_operator(const char c) {
    return c==',' || c==';' || c==':' || c=='(' || c==')' ||
           c=='+' || c=='-' || c=='*' || c=='/' || c=='=' ||
           c=='!';
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

void lexer::skip_blank() {
    while (index<source.length() &&
        (std::isspace(source[index]) ||
        source[index]=='\n' ||
        source[index]=='\r')) {
        ++index;
    }
}

void lexer::gen_number() {
    auto res = std::string("");
    res += source[index];
    ++index;
    while(index<source.length() && std::isdigit(source[index])) {
        res += source[index];
        ++index;
    }
    tokens.push_back({tok_type::tok_num, res});
}

void lexer::gen_identifier() {
    auto res = std::string("");
    res += source[index];
    ++index;
    while(index<source.length() &&
        (is_identifier(source[index]) ||
        std::isdigit(source[index]))) {
        res += source[index];
        ++index;
    }
    if (keyword_type_table()->count(res)) {
        tokens.push_back({keyword_type_table()->at(res), res});
        return;
    }
    tokens.push_back({tok_type::tok_id, res});
}

void lexer::gen_string() {
    auto res = std::string("");
    res += source[index];
    ++index;
    while(index<source.length() && source[index]!='\"') {
        res += source[index];
        ++index;
    }
    if (index>=source.length()) {
        std::cerr << "get eof when generating string.\n";
        return;
    }
    if (source[index]=='\"') {
        res += source[index];
        ++index;
    }
    tokens.push_back({tok_type::tok_str, res});
}

void lexer::gen_single_operator() {
    auto res = std::string("");
    res += source[index];
    ++index;
    if (!operator_type_table()->count(res)) {
        std::cerr << "invalid operator \"" << res << "\"\n";
        return;
    }
    tokens.push_back({operator_type_table()->at(res), res});
}

void lexer::scan(const std::string& filename) {
    readfile(filename);
    if (source.empty()) {
        return;
    }

    index = 0;
    while(index<source.length()) {
        skip_blank();
        if (index>=source.length()) {
            break;
        }

        if (is_number(source[index])) {
            gen_number();
        } else if (is_identifier(source[index])) {
            gen_identifier();
        } else if (is_quote(source[index])) {
            gen_string();
        } else if (is_single_operator(source[index])) {
            gen_single_operator();
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

    tokens.push_back({tok_type::tok_eof, ""});
}

void lexer::dump() const {
    for(const auto& i : tokens) {
        std::clog << i.content << "\n";
    }
}

}
