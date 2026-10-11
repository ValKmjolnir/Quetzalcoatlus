#include "model/qgpt_bin_reader.hpp"

namespace quetzal::gpt {

static std::uint32_t read_u32(std::istream& in) {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("unexpected EOF");
    }
    return v;
}

void weights_manager::read(std::ifstream& in) {
    const std::uint32_t tensor_count = read_u32(in);

    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        std::cout << "[Info] weights_manager: loading tensor "
                  << i + 1 << "/" << tensor_count
                  << " (" << int((i + 1) * 100.f / tensor_count) << "%)\r"
                  << std::flush;
        // read tensor name
        std::string name;
        const std::uint32_t name_len = read_u32(in);
        name.resize(name_len);
        in.read(name.data(), static_cast<std::streamsize>(name_len));
        if (!in) {
            throw std::runtime_error("truncated name");
        }

        // read shape
        std::vector<std::size_t> shape;
        const std::uint32_t ndim = read_u32(in);
        shape.resize(ndim);
        std::size_t numel = 1;
        for (std::uint32_t d = 0; d < ndim; ++d) {
            shape[d] = read_u32(in);
            numel *= shape[d];
        }

        // read dtype enum
        const std::uint32_t dtype = read_u32(in);
        if (static_cast<weight_dtype>(dtype) != weight_dtype::float32) {
            throw std::runtime_error("unsupported dtype " + std::to_string(dtype) +
                                     " for tensor " + name);
        }

        tensor::tensor<float> t(shape);

        // read tensor buffer
        in.read(reinterpret_cast<char*>(t.data()),
                static_cast<std::streamsize>(numel * sizeof(float)));
        if (!in) {
            throw std::runtime_error("truncated data for " + name);
        }

        weights_.emplace(name, std::move(t));
    }
    std::cout << std::endl;
}

void bin_reader::read(std::ifstream& in) {
    // read added token list
    auto added_tokens_len = read_u32(in);
    for (std::uint32_t i = 0; i < added_tokens_len; ++i) {
        auto token_len = read_u32(in);
        std::string token;
        token.resize(token_len);
        in.read(token.data(), static_cast<std::streamsize>(token_len));
        if (!in) {
            throw std::runtime_error("truncated added token");
        }

        special_vocab.push_back(token);
    }

    // read vocab
    auto vocab_size = read_u32(in);
    for (std::uint32_t i = 0; i < vocab_size; ++i) {
        auto token_index = read_u32(in);
        auto token_len = read_u32(in);
        std::string token;
        token.resize(token_len);
        in.read(token.data(), static_cast<std::streamsize>(token_len));
        if (!in) {
            throw std::runtime_error("truncated token");
        }

        vocab.push_back(token);
        vocab_index.insert(token, token_index);
    }

    // read merges
    auto merges_size = read_u32(in);
    for (std::uint32_t i = 0; i < merges_size; ++i) {
        auto token_a_len = read_u32(in);
        std::string token_a;
        token_a.resize(token_a_len);
        in.read(token_a.data(), static_cast<std::streamsize>(token_a_len));
        if (!in) {
            throw std::runtime_error("truncated token");
        }

        auto token_b_len = read_u32(in);
        std::string token_b;
        token_b.resize(token_b_len);
        in.read(token_b.data(), static_cast<std::streamsize>(token_b_len));
        if (!in) {
            throw std::runtime_error("truncated token");
        }

        merge_pairs.push_back(bbpe::index_pair {
            vocab_index.at(token_a),
            vocab_index.at(token_b)
        });
    }
}

qgpt_bin_reader::qgpt_bin_reader(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "QGPT") {
        throw std::runtime_error("not a QGPT model file: " + path);
    }

    weights_manager_.read(in);
    bin_reader_.read(in);
}

}
