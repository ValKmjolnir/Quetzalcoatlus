#include "lexer.h"

#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc<2) {
        return 0;
    }
    quetza::lexer lexer;
    lexer.scan(argv[1]);
    return 0;
}
