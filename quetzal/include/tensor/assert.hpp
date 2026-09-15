#pragma once

#include <string>
#include <stdexcept>

#define QUETZAL_ASSERT(cond, msg)\
    do {                                                               \
        if (!(cond)) {                                                 \
            throw std::runtime_error(std::string(__FILE__) + ":" +     \
                                     std::to_string(__LINE__) + ": " + \
                                     (msg));                           \
        }                                                              \
    } while (0)
