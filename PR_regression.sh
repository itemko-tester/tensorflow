#!/bin/bash

#-----------
# configure
#-----------
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_S3=0
export TF_NEED_VERBS=1
PYTHON_BIN_PATH=$(which python || which python3 || true)
yes "" | $PYTHON_BIN_PATH configure.py
#-----------
# deploy
#-----------
export TENSORFLOW_HOME="/var/jenkins/workspace/Mellanox_Tensorflow_benchmark" 
cd /var/jenkins/workspace/benchmarks/scripts/tf_cnn_benchmarks
./deploy.sh -s -c -v 33333 1 1 192.168.10.14 192.168.10.15

