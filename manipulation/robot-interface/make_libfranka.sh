cd libfranka

# Change built test to Off
sed -i '129s/.*/option(BUILD_TESTS "Build tests" OFF)/' CMakeLists.txt

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make

cd ..