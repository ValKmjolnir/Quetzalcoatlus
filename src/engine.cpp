#include "engine.h"

#include <iostream>

namespace quetza {

void engine::test() const {
    table relation = {};
    relation.insert({1, 2, 3});
    relation.insert({1, 2, 2});
    relation.insert({2, 2, 3});
    relation.insert({1, 2, 3});
    relation.insert({1, 2, 2});
    relation.insert({2, 2, 3});

    for(const auto& i : relation) {
        for(const auto& j : i) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }

    std::cout << "--------------\n";

    relation.erase({1, 2, 3});

    for(const auto& i : relation) {
        for(const auto& j : i) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }

    std::cout << "--------------\n";

    relation.erase({2, 2, 3});

    for(const auto& i : relation) {
        for(const auto& j : i) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }

    std::cout << "--------------\n";

    relation.erase({1, 2, 2});

    for(const auto& i : relation) {
        for(const auto& j : i) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }

    std::cout << "--------------\n";
}

}