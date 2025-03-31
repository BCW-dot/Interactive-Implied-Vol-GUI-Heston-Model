set(CMAKE_CUDA_COMPILER "/opt/software/easybuild/amd/software/CUDA/12.6.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/opt/software/easybuild/amd/software/GCCcore/13.2.0/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.6.20")
set(CMAKE_CUDA_DEVICE_LINKER "/opt/software/easybuild/amd/software/CUDA/12.6.0/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/opt/software/easybuild/amd/software/CUDA/12.6.0/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "13.2")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/opt/software/easybuild/amd/software/CUDA/12.6.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/opt/software/easybuild/amd/software/CUDA/12.6.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.6.20")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/opt/software/easybuild/amd/software/CUDA/12.6.0")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "80-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/opt/software/easybuild/amd/software/CUDA/12.6.0/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/opt/software/easybuild/amd/software/CUDA/12.6.0/targets/x86_64-linux/lib/stubs;/opt/software/easybuild/amd/software/CUDA/12.6.0/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/opt/software/easybuild/amd/software/Brotli/1.1.0-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/OpenSSL/1.1/include;/opt/software/easybuild/amd/software/libarchive/3.7.2-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/XZ/5.4.4-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/cURL/8.3.0-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/bzip2/1.0.8-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/ncurses/6.4-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/binutils/2.40-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/zlib/1.2.13-GCCcore-13.2.0/include;/opt/software/easybuild/amd/software/CUDA/12.6.0/nvvm/include;/opt/software/easybuild/amd/software/CUDA/12.6.0/extras/CUPTI/include;/opt/software/easybuild/amd/software/GCCcore/13.2.0/include/c++/13.2.0;/opt/software/easybuild/amd/software/GCCcore/13.2.0/include/c++/13.2.0/x86_64-pc-linux-gnu;/opt/software/easybuild/amd/software/GCCcore/13.2.0/include/c++/13.2.0/backward;/opt/software/easybuild/amd/software/GCCcore/13.2.0/lib/gcc/x86_64-pc-linux-gnu/13.2.0/include;/opt/software/easybuild/amd/software/GCCcore/13.2.0/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/opt/software/easybuild/amd/software/CUDA/12.6.0/targets/x86_64-linux/lib/stubs;/opt/software/easybuild/amd/software/CUDA/12.6.0/targets/x86_64-linux/lib;/opt/software/easybuild/amd/software/Brotli/1.1.0-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/OpenSSL/1.1/lib64;/opt/software/easybuild/amd/software/libarchive/3.7.2-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/XZ/5.4.4-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/cURL/8.3.0-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/bzip2/1.0.8-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/ncurses/6.4-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/binutils/2.40-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/zlib/1.2.13-GCCcore-13.2.0/lib64;/opt/software/easybuild/amd/software/CUDA/12.6.0/stubs/lib64;/opt/software/easybuild/amd/software/GCCcore/13.2.0/lib/gcc/x86_64-pc-linux-gnu/13.2.0;/opt/software/easybuild/amd/software/GCCcore/13.2.0/lib64;/lib64;/usr/lib64;/opt/software/easybuild/amd/software/Brotli/1.1.0-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/OpenSSL/1.1/lib;/opt/software/easybuild/amd/software/libarchive/3.7.2-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/XZ/5.4.4-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/cURL/8.3.0-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/bzip2/1.0.8-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/ncurses/6.4-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/binutils/2.40-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/zlib/1.2.13-GCCcore-13.2.0/lib;/opt/software/easybuild/amd/software/CUDA/12.6.0/lib;/opt/software/easybuild/amd/software/GCCcore/13.2.0/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/opt/software/easybuild/amd/software/binutils/2.40-GCCcore-13.2.0/bin/ld")
set(CMAKE_AR "/opt/software/easybuild/amd/software/binutils/2.40-GCCcore-13.2.0/bin/ar")
set(CMAKE_MT "")
