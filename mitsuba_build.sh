mkdir -p build
cd build
cmake -GNinja .. -DH5_USE_HALF_FLOAT=1
ninja
