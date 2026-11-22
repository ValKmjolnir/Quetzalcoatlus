#include "util/testing.hpp"
#include "util/json/json.hpp"

#include <iostream>

using namespace quetzal;

void test_boolean() {
    {
        std::string test_str = "{\"flag_true\":true,\"flag_false\":false}";
        json::json j;
        auto n = j.parse(test_str);
        CHECK(n->type == json::node_type::object &&
              n->value.obj->count("flag_true") &&
              n->value.obj->count("flag_false") &&
              n->value.obj->at("flag_true")->type == json::node_type::boolean &&
              n->value.obj->at("flag_true")->value.flag == true &&
              n->value.obj->at("flag_false")->type == json::node_type::boolean &&
              n->value.obj->at("flag_false")->value.flag == false
        );
        delete n;
    }
    {
        std::string test_str = "[true,true,false,false,true,false]";
        json::json j;
        auto n = j.parse(test_str);
        CHECK(j.stringify(n) == test_str);
        delete n;
    }
}

int main() {
    test_boolean();
    return test::summary("json_lib_test");
}
