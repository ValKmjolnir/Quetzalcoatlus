#include "weights.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace qgpt {

namespace {

std::uint32_t read_u32(std::istream& in, const std::string& path) {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("unexpected EOF in " + path);
    }
    return v;
}

}  // namespace

std::map<std::string, Tensor> load_weights(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "QGPT") {
        throw std::runtime_error("not a QGPT weights file: " + path);
    }

    const std::uint32_t count = read_u32(in, path);

    std::map<std::string, Tensor> tensors;
    for (std::uint32_t i = 0; i < count; ++i) {
        Tensor t;

        const std::uint32_t name_len = read_u32(in, path);
        t.name.resize(name_len);
        in.read(t.name.data(), static_cast<std::streamsize>(name_len));
        if (!in) {
            throw std::runtime_error("truncated name in " + path);
        }

        const std::uint32_t ndim = read_u32(in, path);
        t.shape.resize(ndim);
        std::size_t numel = 1;
        for (std::uint32_t d = 0; d < ndim; ++d) {
            t.shape[d] = read_u32(in, path);
            numel *= t.shape[d];
        }

        const std::uint32_t dtype = read_u32(in, path);
        if (dtype != 0) {
            throw std::runtime_error("unsupported dtype " + std::to_string(dtype) +
                                     " for tensor " + t.name);
        }

        t.data.resize(numel);
        in.read(reinterpret_cast<char*>(t.data.data()),
                static_cast<std::streamsize>(numel * sizeof(float)));
        if (!in) {
            throw std::runtime_error("truncated data for " + t.name);
        }

        const std::string key = t.name;
        tensors.emplace(key, std::move(t));
    }

    return tensors;
}

}
