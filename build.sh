#!/bin/bash

build_dir=build
build_sys=Ninja

cmake -B $build_dir -G $build_sys && cmake --build build
