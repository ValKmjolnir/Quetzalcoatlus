find_package(OpenMP REQUIRED)

add_library(qgpt-weights STATIC weights.cpp)
target_include_directories(qgpt-weights PUBLIC ${CMAKE_SOURCE_DIR}/deploy)
target_link_libraries(qgpt-weights PUBLIC OpenMP::OpenMP_CXX)

add_executable(qgpt-load main.cpp)
target_link_libraries(qgpt-load qgpt-weights)
