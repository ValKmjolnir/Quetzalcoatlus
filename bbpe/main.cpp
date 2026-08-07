#include "bbpe.hpp"

#include <fstream>

int main() {
    bbpe::BBPE bpe("data/text.txt");

    std::ofstream debug("debug.log");
    bpe.dump(debug);

    std::ofstream json("debug.json");
    bpe.dump_json(json);
    return 0;
}
