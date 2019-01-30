# Copy needed files from libfranka cmake. -n means don't copy if dest. file exists
[ -d cmake ] || mkdir cmake
cp -n libfranka/cmake/FindEigen3.cmake cmake/
cp -n libfranka/cmake/FindPoco.cmake cmake/

# Get CPU core count
n_cores=grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}' 

[ -d build ] || mkdir build
cd build
cmake -j$n_cores -DCMAKE_BUILD_TYPE=Release .. && make
cd ..