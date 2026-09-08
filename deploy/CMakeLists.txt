find_package(OpenMP)

add_library(quetzal STATIC
            lib/weights.cpp
            lib/tensor.cpp)
target_include_directories(quetzal PUBLIC ${CMAKE_SOURCE_DIR}/deploy/include)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(quetzal PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(quetzal PUBLIC QGPT_USE_OPENMP)
else()
    message(STATUS "OpenMP not found")
endif()

add_executable(qgpt-load main.cpp)
target_link_libraries(qgpt-load quetzal)

add_executable(tensor_test test/tensor_test.cpp)
target_link_libraries(tensor_test quetzal)
