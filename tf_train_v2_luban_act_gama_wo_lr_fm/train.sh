#!/bin/bash

source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf
ckpt_day=$train_end_day
: ${is_auto_train:=0}
if [ $is_auto_train -eq 1 ];then
    current_date=$(date +%Y%m%d)
    #current_date=20260504
    temp_date="$current_date"
    max_days=20
    found=0
    eday=""
    bday=$(date -d "$train_end_day +1 day" +%Y%m%d)
    count=0

    while [[ "$temp_date" -ge "$bday" ]]; do
        if [ $count -ge $max_days ]; then
            echo "例行时间太久"
            exit 1
        fi

        done_file_path=${train_hdfs_dir}/${temp_date}"/_SUCCESS"
        if $hadoop fs -test -e "$done_file_path"; then
            echo "temp_date is exist"
            eday="$temp_date"
            found=1
            break
        fi

        temp_date=$(date -d "$temp_date -1 day" +%Y%m%d)
        if [[ $? -ne 0 ]]; then
            echo "错误: 日期计算失败"
            exit 1
        fi

        count=$((count + 1))
    done

    if [[ $found -eq 0 ]]; then
        echo "no ready day"
        exit 1
    fi

    if [[ ! -d "./model/checkpoints/${ckpt_day}_0" ]]; then
        echo "ckpt dir not exist"
        exit 1
    fi

    train_start_day=$bday
    train_end_day=$eday
fi

exec 1>"./log/train_log_${train_end_day}_$nowt" 2>&1

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u train.py -data $train_hdfs_dir -start_day $bday -end_day $eday

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
