#include "util/json/json.hpp"

#include <string>
#include <sstream>
#include <cmath>
#include <utility>

namespace quetzal::json {

static std::string char_to_hex(const char c) {
    const char hextbl[] = "0123456789abcdef";
    return { hextbl[(c & 0xf0) >> 4], hextbl[c & 0x0f] };
}

double json::parse_num(const std::string& n) {
    char* end = nullptr;
    errno = 0;
    double res = std::strtod(n.data(), &end);
    if (end == n.data() || *end != '\0' || errno == ERANGE || !std::isfinite(res)) {
        error_info() += "json::parse_num: unacceptable number\n";
    }
    return res;
}

std::string json::var_generate(node* n) {
    switch (n->type) {
        case node_type::null: return "null";
        case node_type::boolean:
            return n->value.flag ? "true" : "false";
        case node_type::number: {
            std::stringstream out;
            out << n->value.num;
            if (std::isnan(n->value.num)) {
                error_info() += "json::stringify: cannot generate number nan\n";
            }
            if (std::isinf(n->value.num)) {
                error_info() += "json::stringify: cannot generate number inf\n";
            }
            return out.str();
        }
        case node_type::string:
            return "\"" + *n->value.str + "\"";
        case node_type::array:
            return arr_generate(n);
        case node_type::object:
            return obj_generate(n);
    }

    return "undefined";
}

std::string json::arr_generate(node* n) {
    std::string out = "[";
    for (auto& i : *n->value.arr) {
        out += var_generate(i) + ",";
    }
    if (out.back() == ',') {
        out.pop_back();
    }
    out += "]";
    return out;
}

std::string json::obj_generate(node* n) {
    std::string out = "{";
    for (auto& i : *n->value.obj) {
        out += "\"" + i.first + "\":";
        out += var_generate(i.second) + ",";
    }
    if (out.back() == ',') {
        out.pop_back();
    }
    out += "}";
    return out;
}

std::string json::stringify(node* n) {
    if (n == nullptr) {
        error_info() += "json::stringify: get nullptr\n";
        return "";
    }

    error_info() = "";
    if (n->type == node_type::array) {
        return arr_generate(n);
    } else if (n->type == node_type::object) {
        return obj_generate(n);
    }
    return "[]";
}

void json::next() {
    while (ptr < text.length() && !check(text[ptr])) {
        if (text[ptr] == '\n') {
            ++line;
        } else if (text[ptr] != ' ' && text[ptr] != '\t' && text[ptr] != '\r') {
            error_info() += "json::parse: line " + std::to_string(line);
            error_info() += ": invalid character `0x";
            error_info() += char_to_hex(text[ptr]);
            error_info() += "`\n";
        }
        ++ptr;
    }
    if (ptr >= text.length()) {
        this_token = {token_type::tok_eof, "eof"};
        return;
    }
    auto c = text[ptr];
    switch (c) {
        case '{': this_token = {token_type::tok_lbrace, "{"}; ++ptr; return;
        case '}': this_token = {token_type::tok_rbrace, "}"}; ++ptr; return;
        case '[': this_token = {token_type::tok_lbrkt, "["}; ++ptr; return;
        case ']': this_token = {token_type::tok_rbrkt, "]"}; ++ptr; return;
        case ',': this_token = {token_type::tok_comma, ","}; ++ptr; return;
        case ':': this_token = {token_type::tok_colon, ":"}; ++ptr; return;
        default: break;
    }
    if (is_num(c) || c == '-' || c == '+') {
        auto temp = std::string(1, c);
        ++ptr;
        while (ptr < text.length() && (
            is_num(text[ptr]) ||
            text[ptr] == '.' ||
            text[ptr] == 'e' ||
            text[ptr] == '-' ||
            text[ptr] == '+')) {
            temp += text[ptr];
            ++ptr;
        }
        --ptr;
        this_token = {token_type::tok_num, temp};
    } else if (is_id(c)) {
        auto temp = std::string(1, c);
        ++ptr;
        while (ptr < text.length() && (is_id(text[ptr]) || is_num(text[ptr]))) {
            temp += text[ptr];
            ++ptr;
        }
        --ptr;
        if (temp == "true" || temp == "false") {
            this_token = {token_type::tok_bool, temp};
        } else if (temp == "null") {
            this_token = {token_type::tok_null, temp};
        } else {
            this_token = {token_type::tok_id, temp};
        }
    } else if (c == '"' || c == '\'') {
        auto begin = c;
        auto temp = std::string("");
        ++ptr;
        while (ptr < text.length() && text[ptr] != begin) {
            temp += text[ptr];
            ++ptr;
            if (text[ptr-1] == '\\' && ptr < text.length()) {
                temp += text[ptr];
                ++ptr;
            }
        }
        this_token = {token_type::tok_str, temp};
    }
    ++ptr;
    return;
}

void json::match(token_type type) {
    if (this_token.type != type) {
        error_info() += "json::parse: line " + std::to_string(line);
        error_info() += ": expect " + get_type_content(type) + " but get `";
        error_info() += this_token.content + "`.\n";
    }
    next();
    return;
}

void json::arr_member(node* n) {
    auto& vec = *n->value.arr;
    if (this_token.type == token_type::tok_lbrace) {
        vec.push_back(obj_elem_gen());
    } else if (this_token.type == token_type::tok_lbrkt) {
        vec.push_back(arr_elem_gen());
    } else if (this_token.type == token_type::tok_str) {
        vec.push_back(node::str(this_token.content));
        next();
    } else if (this_token.type == token_type::tok_bool) {
        vec.push_back(node::boolean(this_token.content == "true"));
        next();
    } else if (this_token.type == token_type::tok_null) {
        vec.push_back(node::null());
        next();
    } else if (this_token.type == token_type::tok_num) {
        vec.push_back(node::num(parse_num(this_token.content)));
        next();
    }
}

node* json::arr_elem_gen() {
    node* n = new node(node_type::array);
    match(token_type::tok_lbrkt);
    arr_member(n);
    while (this_token.type == token_type::tok_comma) {
        match(token_type::tok_comma);
        arr_member(n);
    }
    match(token_type::tok_rbrkt);
    return n;
}

void json::obj_member(node* n) {
    auto& obj = *n->value.obj;
    const auto name = this_token.content;
    if (this_token.type == token_type::tok_rbrace) {
        return;
    }
    if (this_token.type == token_type::tok_str) {
        match(token_type::tok_str);
    } else {
        match(token_type::tok_id);
    }
    match(token_type::tok_colon);
    if (this_token.type == token_type::tok_lbrace) {
        obj.insert({name, obj_elem_gen()});
    } else if (this_token.type == token_type::tok_lbrkt) {
        obj.insert({name, arr_elem_gen()});
    } else if (this_token.type == token_type::tok_str) {
        obj.insert({name, node::str(this_token.content)});
        next();
    } else if (this_token.type == token_type::tok_bool) {
        obj.insert({name, node::boolean(this_token.content == "true")});
        next();
    } else if (this_token.type == token_type::tok_null) {
        obj.insert({name, node::null()});
        next();
    } else if (this_token.type == token_type::tok_num) {
        obj.insert({name, node::num(parse_num(this_token.content))});
        next();
    }
}

node* json::obj_elem_gen() {
    auto n = new node(node_type::object);
    match(token_type::tok_lbrace);
    obj_member(n);
    while (this_token.type == token_type::tok_comma) {
        match(token_type::tok_comma);
        obj_member(n);
    }
    match(token_type::tok_rbrace);
    return n;
}

void json::check_eof() {
    next();
    if (this_token.type == token_type::tok_eof) {
        return;
    }
    while (this_token.type != token_type::tok_eof) {
        error_info() += "json::parse: line " + std::to_string(line);
        error_info() += ": expect " + get_type_content(token_type::tok_eof);
        error_info() += " but get `" + this_token.content + "`.\n";
        next();
    }
}

bool parse_result::ok() const {
    return root != nullptr;
}

parse_result json::parse(const std::string& input) {
    line = 1;
    ptr = 0;
    this_token = {token_type::tok_eof, ""};
    error_info() = "";

    if (input.empty()) {
        error_info() += "json::parse: empty string.\n";
        return {nullptr, std::move(info)};
    }
    text = input;
    next();

    std::unique_ptr<node> root;
    if (this_token.type == token_type::tok_lbrkt) {
        root.reset(arr_elem_gen());
    } else {
        root.reset(obj_elem_gen());
    }
    check_eof();
    if (!info.empty()) {
        root.reset();
    }
    return {std::move(root), std::move(info)};
}

}
