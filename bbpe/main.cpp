#include "bbpe.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: bbpe <input text>\n";
        return -1;
    }

    const std::vector<std::string> special = {
        "<|pad|>", "<|unk|>", "<|endoftext|>",
        "<|im_start|>", "<|im_end|>"
    };

    bbpe::BBPE bpe(special);
    bpe.merge(argv[1]);

    std::ofstream debug("debug.log");
    bpe.dump(debug);

    std::ofstream json("debug.json");
    bpe.dump_json(json);
    return 0;
}
