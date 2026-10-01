#pragma once

#ifdef _OPENMP
#include <omp.h>
#define QGPT_STR_(x) #x
#define QGPT_STR(x) QGPT_STR_(x)
#define OMP_FOR _Pragma("omp parallel for")
#define OMP_FOR_IF(cond) _Pragma(QGPT_STR(omp parallel for if(cond)))
#define OMP_FOR_SCHED _Pragma("omp parallel for schedule(static)")
#define OMP_FOR_COLLAPSE2_SCHED _Pragma("omp parallel for collapse(2) schedule(static)")
#define OMP_MAX_THREADS() omp_get_num_threads()
#else
#define OMP_FOR
#define OMP_FOR_IF(cond)
#define OMP_FOR_SCHED
#define OMP_FOR_COLLAPSE2_SCHED
#define OMP_MAX_THREADS() 1
#endif

#include <cstdint>
#include <cstring>

namespace quetzal::tensor {
inline constexpr std::size_t omp_min_elems = 32768;
inline constexpr std::size_t omp_min_macs  = 524288;
}

