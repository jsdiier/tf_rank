#!/bin/bash

source /etc/profile
source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf

exec 1>"./log/test_log_${test_end_day}_$nowt" 2>&1

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u test.py -data $train_hdfs_dir -start_day $test_start_day -end_day $test_end_day
