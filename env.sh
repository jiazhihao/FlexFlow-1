#! /usr/bin/env bash

ml restore flexflow

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export FF_HOME="$DIR"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/FlexFlow/protobuf/src/.libs/"
export GASNET_BACKTRACE=1
export DEBUG=0
export GASNET="$HOME/FlexFlow/gasnet/release"
export USE_GASNET=1
export LD_FLAGS="-lcudnn -lpmix ${LD_FLAGS:-} -L$HOME/FlexFlow/protobuf/src/.libs/ -lcuda -lcurand -lcublas -lprotobuf"
export USE_CUDA=1
export USE_HDF=1
export CMAKE_ARGS="-DProtobuf_INCLUDE_DIR='/home/users/unger/FlexFlow/protobuf/src/' -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF"
export PATH="$HOME/FlexFlow/protobuf/src/.libs/:$PATH"
