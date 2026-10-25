#include "util/timestamp.hpp"

namespace quetzal::util {

std::string format_time(std::time_t t) {
    std::tm tm {};
#ifndef _WIN32
    localtime_r(&t, &tm);
#else
    localtime_s(&tm, &t);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tm);
    return buf;
}

}