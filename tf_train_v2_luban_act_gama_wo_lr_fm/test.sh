#!/bin/bash

source /etc/profile
source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf

#exec 1>"./log/test_log_${test_end_day}_$nowt" 2>&1

if [ -z "$day_list" ];then
    bt=`date -d"$test_start_day" +"%s"`
    et=`date -d"$test_end_day" +"%s"`
    bday=`date -d"$test_start_day" +%Y%m%d`
    eday=`date -d"$test_end_day" +%Y%m%d`

    day_span=`echo "($et-$bt)/(24*3600)" | bc`
    for((i=$day_span;i>0;i--));do
        day=`date +"%Y%m%d" -d "$i day ago $test_end_day"`
        day_list=$day_list""$day" "
    done
    day=`date +"%Y%m%d" -d "$test_end_day"`
    day_list=$day_list""$day
fi

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u test.py -data $test_hdfs_dir -start_day $test_start_day -end_day $test_end_day
