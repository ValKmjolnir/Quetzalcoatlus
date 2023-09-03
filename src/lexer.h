#pragma once

#include <cstring>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>

namespace quetza {

enum class tok_type {
    tok_ull,
    tok_id,
    tok_num,
    tok_str,
    tok_comma,
    tok_semi,
    tok_colon,
    tok_lcurve,
    tok_rcurve,
    tok_add,
    tok_sub,
    tok_mult,
    tok_div,
    tok_equal,
    tok_not,
    tok_eof
};

struct token {
    tok_type type;
    std::string content;
};

class lexer {
private:
    std::string source = "";
    std::vector<token> tokens = {};
    size_t index;

private:
    static const std::unordered_map<std::string, tok_type>* operator_type_table() {
        static const std::unordered_map<std::string, tok_type> operator_table = {
            {",", tok_type::tok_comma},
            {";", tok_type::tok_semi},
            {":", tok_type::tok_colon},
            {"(", tok_type::tok_lcurve},
            {")", tok_type::tok_rcurve},
            {"+", tok_type::tok_add},
            {"-", tok_type::tok_sub},
            {"*", tok_type::tok_mult},
            {"/", tok_type::tok_div},
            {"=", tok_type::tok_equal},
            {"!", tok_type::tok_not}
        };
        return &operator_table;
    }
    static const std::unordered_map<std::string, tok_type>* keyword_type_table() {
        static const std::unordered_map<std::string, tok_type> keyword_table = {};
        return &keyword_table;
    }

private:
    static constexpr bool is_quote(const char);
    static constexpr bool is_identifier(const char);
    static constexpr bool is_number(const char);
    static constexpr bool is_single_operator(const char);
    void readfile(const std::string&);
    void skip_blank();
    void gen_number();
    void gen_identifier();
    void gen_string();
    void gen_single_operator();

public:
    void scan(const std::string&);
    void dump() const;
};

}
