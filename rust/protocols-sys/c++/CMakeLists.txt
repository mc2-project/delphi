cmake_minimum_required(VERSION 3.12)
project(DELPHI VERSION 1.0 LANGUAGES C CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(LIB "src/lib")
set(BIN "src/bin")
add_library( DelphiOffline STATIC
    ${LIB}/interface.cpp ${LIB}/conv2d.cpp ${LIB}/fc_layer.cpp
    ${LIB}/triples_gen.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/lib)

include(FetchContent)
FetchContent_Declare(
  seal
  GIT_REPOSITORY https://github.com/microsoft/SEAL
  GIT_TAG        v3.6.5
)
option(SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT OFF)
option(SEAL_USE_ZLIB OFF)
option(SEAL_USE_ZSTD ON)
FetchContent_MakeAvailable(seal)

find_package(OpenMP REQUIRED)
target_link_libraries(DelphiOffline seal OpenMP::OpenMP_CXX)


option(UNITTESTS "Build unittests" OFF)
if(UNITTESTS)
    include_directories(${CMAKE_SOURCE_DIR}/src/bin)
    add_executable(unittests ${BIN}/unittests.cpp ${LIB}/im2col.cpp)
    add_executable(benchmark ${BIN}/benchmark.cpp ${LIB}/im2col.cpp)
    add_executable(interface ${BIN}/interface.c)

    # Unittests make use of Eigen
    find_package (Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries (unittests Eigen3::Eigen)
    target_link_libraries (benchmark Eigen3::Eigen)

    target_link_libraries (benchmark DelphiOffline)
    target_link_libraries (unittests DelphiOffline)
    target_link_libraries (interface DelphiOffline)
endif()

install(TARGETS DelphiOffline DESTINATION .)
