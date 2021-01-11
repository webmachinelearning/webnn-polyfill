#!/bin/sh
TAG=android-cts-10.0_r5
NNAPI_VERSION="
V1_0
V1_1
V1_2
"

# download and unzip Specs tarball files
for version in ${NNAPI_VERSION}
do
  wget https://android.googlesource.com/platform/frameworks/ml/+archive/refs/tags/${TAG}/nn/runtime/test/specs/${version}.tar.gz
  mkdir -p specs/${version}
  tar -xvzf ${version}.tar.gz -C specs/${version}
done