#/bin/bash

export ANDROID_NDK=/home/ljx/Android/Sdk/ndk/20b

rm -r build
mkdir build && cd build

cmake  ..

make && make install

