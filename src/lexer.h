#pragma once

#include <cstring>
#include <sstream>
#include <fstream>
#include <vector>

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
    std::string source = "";
    std::vector<token> tokens = {};

private:
    static constexpr bool isquote(const char);
    static constexpr bool isidentifier(const char);
    static constexpr bool isnumber(const char);
    void readfile(const std::string&);

public:
    void scan(const std::string&);
};

}
