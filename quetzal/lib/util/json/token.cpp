#include "util/json/token.hpp"

namespace quetzal::json {

std::string get_type_content(token_type type) {
    switch (type) {
        case token_type::tok_eof: return "eof";
        case token_type::tok_lbrace: return "`{`";
        case token_type::tok_rbrace: return "`}`";
        case token_type::tok_lbrkt: return "`[`";
        case token_type::tok_rbrkt: return "`]`";
        case token_type::tok_comma: return "`,`";
        case token_type::tok_colon: return "`:`";
        case token_type::tok_str: return "string";
        case token_type::tok_num: return "number";
        case token_type::tok_id: return "identifier";
        case token_type::tok_bool: return "boolean";
        case token_type::tok_null: return "null";
    }
    // unreachable
    return "";
}

}
