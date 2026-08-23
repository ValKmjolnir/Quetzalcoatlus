find_package(OpenMP)

add_library(qgpt-weights STATIC weights.cpp)
target_include_directories(qgpt-weights PUBLIC ${CMAKE_SOURCE_DIR}/deploy)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(qgpt-weights PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(qgpt-weights PUBLIC QGPT_USE_OPENMP)
else()
    message(STATUS "OpenMP not found")
endif()

add_executable(qgpt-load main.cpp)
target_link_libraries(qgpt-load qgpt-weights)
