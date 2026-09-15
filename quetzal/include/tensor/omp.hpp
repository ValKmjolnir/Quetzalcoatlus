#pragma once

#ifdef _OPENMP
#include <omp.h>
#define OMP_FOR _Pragma("omp parallel for")
#define OMP_FOR_SCHED _Pragma("omp parallel for schedule(static)")
#define OMP_MAX_THREADS() omp_get_num_threads()
#else
#define OMP_FOR
#define OMP_FOR_SCHED
#define OMP_MAX_THREADS() 1
#endif
