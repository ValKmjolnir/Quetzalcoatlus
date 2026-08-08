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

    std::ofstream log_out("tokenizer.log");
    bpe.dump(log_out);

    std::ofstream json_out("tokenizer.json");
    bpe.dump_json(json_out);
    return 0;
}
