#pragma once

#include <cstring>
#include <sstream>
#include <fstream>

namespace quetza {

enum class tok_type {
    null,
    id,
    num,
    str,
    comma,
    eof
};

struct token {
    tok_type type;
    std::string content;
};

class lexer {
private:
    std::string source;

private:
    void readfile(const std::string&);

public:
    void scan(const std::string&);
};

}
