#include "lexer.h"
#include "engine.h"

#include <iostream>

#define QUETZA_VERSION "0.0.1"

void info() {
    std::clog << "Quetzalcoatlus v" << QUETZA_VERSION;
    std::clog << " (" << __DATE__ << " " << __TIME__ << ")\n";
}

int main(int argc, const char* argv[]) {
    if (argc<2) {
        info();
        return 0;
    }
    quetza::lexer lexer;
    lexer.scan(argv[1]);
    lexer.dump();
    quetza::engine engine;
    engine.test();
    return 0;
}
