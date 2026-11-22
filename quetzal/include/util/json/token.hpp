#pragma once

#include <string>

namespace quetzal::json {

enum class token_type {
    tok_eof,
    tok_lbrace,
    tok_rbrace,
    tok_lbrkt,
    tok_rbrkt,
    tok_comma,
    tok_colon,
    tok_str,
    tok_num,
    tok_id,
    tok_bool
};

std::string get_type_content(token_type type);

struct token {
    token_type type;
    std::string content;
};

}
