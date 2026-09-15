#include "tensor/weights_manager.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace quetzal {
static std::uint32_t read_u32(std::istream& in, const std::string& path) {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("unexpected EOF in " + path);
    }
    return v;
}

weights_manager::weights_manager(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "QGPT") {
        throw std::runtime_error("not a QGPT weights file: " + path);
    }

    const std::uint32_t tensor_count = read_u32(in, path);

    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        // read tensor name
        std::string name;
        const std::uint32_t name_len = read_u32(in, path);
        name.resize(name_len);
        in.read(name.data(), static_cast<std::streamsize>(name_len));
        if (!in) {
            throw std::runtime_error("truncated name in " + path);
        }

        // read shape
        std::vector<std::size_t> shape;
        const std::uint32_t ndim = read_u32(in, path);
        shape.resize(ndim);
        std::size_t numel = 1;
        for (std::uint32_t d = 0; d < ndim; ++d) {
            shape[d] = read_u32(in, path);
            numel *= shape[d];
        }

        // read dtype enum
        const std::uint32_t dtype = read_u32(in, path);
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
}

}
