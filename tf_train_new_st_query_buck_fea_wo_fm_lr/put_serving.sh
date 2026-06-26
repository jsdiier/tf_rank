#!/bin/bash

source ./common.conf
nowt=`date +"%Y%m%d%H%M"`
CONF_FILE=./common.conf

if [ $# -gt 0 ]; then
    serving_day=$1
else
    serving_day=$train_end_day"00"
fi

exec 1>"./log/"$0".log_${serving_day}_$nowt" 2>&1

dir=./serving_model_0
serving_model_path=$dir/${serving_day}/
if [ $need_dump_serving_model -eq 1 ];then
    mkdir ${serving_model_path}"/assets.extra"
    cp tf_serving_warmup_requests ${serving_model_path}"/assets.extra"

    cd $dir
    tar -czvf ${serving_day}.tar.gz ${serving_day}
    md5sum ${serving_day}.tar.gz > ${serving_day}.tar.gz.md5
    cd -

    $hadoop fs -test -e ${hadoop_serving_model}
    if [ $? -ne 0 ];then
        $hadoop fs -mkdir ${hadoop_serving_model}
    fi

    $hadoop fs -rm -r ${hadoop_serving_model}/${serving_day}*
    $hadoop fs -put $dir/${serving_day}.tar.gz ${hadoop_serving_model}
    $hadoop fs -put $dir/${serving_day}.tar.gz.md5 ${hadoop_serving_model}

    echo ${serving_day} >> ${dir}/serving_model_day
fi
