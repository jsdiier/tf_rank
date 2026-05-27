#!/bin/bash

source /home/luban/.bash_profile
source /etc/profile
source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

LOCK_FILE="my_job.lock"

if [ -f "$LOCK_FILE" ]; then
    echo "Another instance is running, exiting."
    exit 1
fi

echo $$ > "$LOCK_FILE"

cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT

echo "Start job..."

CONF_FILE=./common.conf
ckpt_day=$train_end_day
: ${is_auto_train:=0}
if [ $is_auto_train -eq 1 ];then
    current_date=$(date +%Y%m%d)
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

        #done_file_path=${train_hdfs_dir}/${temp_date}".done"
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

API_URL="http://10.14.127.44:80/api/v3/jobs"
PROJECT_UUID="1b108a55ac6c4893bb427c6efa0a80d5"
TOKEN="185abf3ee9c8454ab21f534ec9f25330"
HEADERS=(
  -H 'Content-Type: application/json'
  -H "JIANSHU-PROJECT-TOKEN: ${TOKEN}"
  -H "JIANSHU-PROJECT-UUID: ${PROJECT_UUID}"
)

echo "正在提交任务..."
RESPONSE=$(curl -s -X POST "${API_URL}" "${HEADERS[@]}" -d "{
    \"userUuid\": \"${USER_UUID}\",
    \"projectUuid\": \"${PROJECT_UUID}\",
    \"imageUuid\": \"e1389ab30bb145d1b72b77f738310a11\",
    \"scriptPath\": \"${SCRIPT_PATH}\",
    \"scriptSourceType\": \"file\",
    \"resourceUuid\": \"${RESOURCE_UUID}\",
    \"regionName\": \"nmg01\",
    \"name\": \"${JOB_NAME}\",
    \"level\": \"PRO\",
    \"backoffLimit\": 1,
    \"priority\": ${PRIORITY},
    \"volumeRegions\":[]
}")

JOB_UUID=$(echo $RESPONSE | jq -r '.data.appId')
if [[ "$JOB_UUID" == "null" || -z "$JOB_UUID" ]]; then
    echo "提交失败，无法获取 Job UUID"
    echo "服务器返回: $RESPONSE"
    exit 1
fi

while true; do
    sleep 60

    QUERY_URL="${API_URL}/${JOB_UUID}?userUuid=${USER_UUID}"
    STATUS_JSON=$(curl -s -X GET "$QUERY_URL" "${HEADERS[@]}")
    RAW_STATUS=$(echo $STATUS_JSON | jq -r '.data.status')

    STATUS=$(echo "$RAW_STATUS" | tr '[:lower:]' '[:upper:]')

    if [[ "$STATUS" == "SUCCEEDED" || "$STATUS" == "COMPLETED" || "$STATUS" == "SUCCESS" ]]; then
        echo "success"
        break
    elif [[ "$STATUS" == "FAILED" || "$STATUS" == "ERROR" || "$STATUS" == "STOPPED" ]]; then
        echo "failed"
        exit 1
    elif [[ "$STATUS" == "null" ]]; then
        echo "error status"
    fi
done

echo "Job done."