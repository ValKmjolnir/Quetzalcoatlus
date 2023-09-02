#include "lexer.h"

#include <iostream>

namespace quetza {

void lexer::readfile(const std::string& filename) {
    std::stringstream ss;
    std::ifstream in(filename, std::ios::binary);
    if (in.fail()) {
        std::cerr << "cannot open file <" << filename << ">\n";
        return;
    }
    ss << in.rdbuf();
    source = ss.str();
}

void lexer::scan(const std::string& filename) {
    readfile(filename);
    std::cout << source;
}

}
