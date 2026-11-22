#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace quetzal::json {

enum class node_type {
    null,
    boolean,
    number,
    string,
    array,
    object,
};

struct node {
    node_type type = node_type::null;
    union {
        bool flag;
        double num;
        std::string* str;
        std::vector<node*>* arr;
        std::unordered_map<std::string, node*>* obj;
    } value;

    node(node_type);
    ~node();

    static node* null() { return new node(node_type::null); }
    static node* boolean(bool f) {
        auto n = new node(node_type::boolean);
        n->value.flag = f;
        return n;
    }
    static node* num(double d) {
        auto n = new node(node_type::number);
        n->value.num = d;
        return n;
    }
    static node* str(const std::string& s) {
        auto n = new node(node_type::string);
        *n->value.str = s;
        return n;
    }
};

}
