#/bin/bash

rm -rf build
meson build -Dpytorch-support=enabled -Denable-pytorch-use-gpu=true
ninja -C build
sudo ninja -C build install
