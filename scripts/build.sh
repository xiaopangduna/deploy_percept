rm -rf build && mkdir build && cd build && cmake .. && make
# rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/x86_64-toolchain.cmake && make