#include "util/json/node.hpp"

namespace quetzal::json {

node::node(node_type t): type(t) {
    switch (type) {
        case node_type::null:
        case node_type::boolean:
        case node_type::number: break;
        case node_type::string:
            value.str = new std::string;
            break;
        case node_type::array:
            value.arr = new std::vector<node*>;
            break;
        case node_type::object:
            value.obj = new std::unordered_map<std::string, node*>;
            break;
    }
}

node::~node() {
    switch (type) {
        case node_type::null:
        case node_type::boolean:
        case node_type::number: break;
        case node_type::string: delete value.str; break;
        case node_type::array: {
            for (auto n: *value.arr) {
                delete n;
            }
            delete value.arr;
        } break;
        case node_type::object: {
            for (auto n: *value.obj) {
                delete n.second;
            }
            delete value.obj;
        } break;
    }
}

}
