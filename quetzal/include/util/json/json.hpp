#pragma once

#include "util/json/token.hpp"
#include "util/json/node.hpp"

#include <string>

namespace quetzal::json {

class json {
private:
    std::string text = "";
    std::size_t line = 1;
    std::size_t ptr = 0;
    token this_token;
    std::string info = "";

private:
    std::string var_generate(node*);
    std::string arr_generate(node*);
    std::string obj_generate(node*);

private:
    bool is_num(char c) {
        return std::isdigit(c);
    }
    bool is_id(char c) {
        return std::isalpha(c) || c=='_';
    }
    bool check(char c) {
        return c=='{' || c=='}' || c=='[' || c==']' ||
               c==':' || c==',' || c=='"' || c=='\'' ||
               c=='-' || c=='+' || is_num(c) || is_id(c);
    }
    void next();
    void match(token_type);
    double parse_num(const std::string&);
    void arr_member(node*);
    node* arr_elem_gen();
    void obj_member(node*);
    node* obj_elem_gen();
    void check_eof();
    std::string& error_info() {
        return info;
    }

public:
    std::string stringify(node*);
    node* parse(const std::string&);
    const std::string& get_error() { return error_info(); }
};

}
