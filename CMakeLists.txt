cmake_minimum_required(VERSION 3.23)
project(sbase_lab_2)

set(CMAKE_CXX_STANDARD 20)

# Пути к заголовочным файлам Intel MKL
set(MKL_INCLUDE_PATH /opt/intel/oneapi/mkl/latest/include)

# Пути к библиотекам Intel MKL
set(MKL_LIBRARY_PATH /opt/intel/oneapi/mkl/latest/lib)

# Добавление заголовочных файлов
include_directories(${MKL_INCLUDE_PATH})

# Добавление библиотек
link_directories(${MKL_LIBRARY_PATH})
link_libraries(mkl_intel_lp64 mkl_sequential mkl_core)

# Поддержка OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(sbase_lab_2 main.cpp kernel.cl)

# Линковка с библиотекой OpenCL
target_link_libraries(sbase_lab_2 ${OpenCL_LIBRARY})