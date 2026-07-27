#!/bin/bash

source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u train.py -data $train_hdfs_dir -start_day 20260615 -end_day 20260615

if [ $? -eq 0 ]; then
    if [ $is_auto_train -eq 1 ];then
        echo "train succ"
        if grep -q "^train_start_day=" "$CONF_FILE"; then
            sed -i "s/^train_start_day=.*/train_start_day=$bday/" "$CONF_FILE"
        else
            echo "train_start_day=$bday" >> "$CONF_FILE"
        fi

        if grep -q "^train_end_day=" "$CONF_FILE"; then
            sed -i "s/^train_end_day=.*/train_end_day=$eday/" "$CONF_FILE"
        else
            echo "train_end_day=$eday" >> "$CONF_FILE"
        fi
    fi
    bash put_serving.sh
else
     echo "train failed"
fi
