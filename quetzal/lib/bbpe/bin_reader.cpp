#include "bbpe/bin_reader.hpp"

#include <fstream>

namespace quetzal::bbpe {

static std::uint32_t read_u32(std::istream& in, const std::string& path) {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("unexpected EOF in " + path);
    }
    return v;
}

bin_reader::bin_reader(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "QTOK") {
        throw std::runtime_error("not a QTOK tokenizer file: " + path);
    }

    // read added token list
    auto added_tokens_len = read_u32(in, path);
    for (std::uint32_t i = 0; i < added_tokens_len; ++i) {
        auto token_len = read_u32(in, path);
        std::string token;
        token.resize(token_len);
        in.read(token.data(), static_cast<std::streamsize>(token_len));
        if (!in) {
            throw std::runtime_error("truncated added token in " + path);
        }

        special_vocab.push_back(token);
    }

    // read vocab
    auto vocab_size = read_u32(in, path);
    for (std::uint32_t i = 0; i < vocab_size; ++i) {
        auto token_index = read_u32(in, path);
        auto token_len = read_u32(in, path);
        std::string token;
        token.resize(token_len);
        in.read(token.data(), static_cast<std::streamsize>(token_len));
        if (!in) {
            throw std::runtime_error("truncated token in " + path);
        }

        vocab.push_back(token);
        vocab_index.insert(token, token_index);
    }

    // read merges
    auto merges_size = read_u32(in, path);
    for (std::uint32_t i = 0; i < merges_size; ++i) {
        auto token_a_len = read_u32(in, path);
        std::string token_a;
        token_a.resize(token_a_len);
        in.read(token_a.data(), static_cast<std::streamsize>(token_a_len));
        if (!in) {
            throw std::runtime_error("truncated token in " + path);
        }

        auto token_b_len = read_u32(in, path);
        std::string token_b;
        token_b.resize(token_b_len);
        in.read(token_b.data(), static_cast<std::streamsize>(token_b_len));
        if (!in) {
            throw std::runtime_error("truncated token in " + path);
        }

        merge_pairs.push_back(index_pair {
            vocab_index.at(token_a),
            vocab_index.at(token_b)
        });
    }
}

}
