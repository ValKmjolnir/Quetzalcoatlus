#include "util/testing.hpp"
#include "util/json/json.hpp"

#include <iostream>

using namespace quetzal;

void test_boolean() {
    {
        std::string test_str = "{\"flag_true\":true,\"flag_false\":false}";
        json::json j;
        auto r = j.parse(test_str);
        CHECK(r.ok() &&
              r.root->type == json::node_type::object &&
              r.root->value.obj->count("flag_true") &&
              r.root->value.obj->count("flag_false") &&
              r.root->value.obj->at("flag_true")->type == json::node_type::boolean &&
              r.root->value.obj->at("flag_true")->value.flag == true &&
              r.root->value.obj->at("flag_false")->type == json::node_type::boolean &&
              r.root->value.obj->at("flag_false")->value.flag == false
        );
    }
    {
        std::string test_str = "[true,true,false,false,true,false]";
        json::json j;
        auto r = j.parse(test_str);
        CHECK(j.stringify(r.root) == test_str);
    }
    {
        json::json j;
        auto r = j.parse("{\"a\":[1,2");
        CHECK(!r.ok() && r.root == nullptr && !r.error.empty());
    }
}

int main() {
    test_boolean();
    return test::summary("json_lib_test");
}
