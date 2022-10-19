#/bin/bash

export ANDROID_NDK=/home/ljx/Android/Sdk/ndk/20b

rm -r build
mkdir build && cd build

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_PLATFORM=android-24 \
    ..

make && make install

