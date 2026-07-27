# coding:utf-8
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import yaml
import tensorflow as tf
import datetime
import time
import gc
from sklearn import metrics
from collections import defaultdict

import warnings
import socket


def auc(lis, scale=1e6):
    def add(d, k, n):
        if k in d:
            d[k] += n
        else:
            d[k] = n

    s = {}
    c = {}
    for x in lis:
        label = 1 if x[0] > 0 else 0
        score = x[1]
        q = int(score * scale)
        add(s, q, 1)
        add(c, q, label)

    k = sorted(s.keys(), reverse=True)
    num_pos = 0
    num_neg = 0
    w = 0.0
    for q in k:
        pos = c[q]
        neg = s[q] - c[q]
        w = w + num_pos * neg + 0.5 * pos * neg
        num_pos += pos
        num_neg += neg
    if num_pos == 0 or num_neg == 0:
        return 0
    return w / (num_pos * num_neg)


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""
    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)
    #
    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        if user_id == 0 or len(user_id) < 10: continue
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag
    impression_total = 0
    total_auc = 0
    #
    u_total_auc = 0.0
    flag_count = 0
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = metrics.roc_auc_score(np.asarray(group_truth[user_id]).astype('float'),
                                        np.asarray(group_score[user_id]).astype('float'))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
            u_total_auc += auc
            flag_count += 1
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    u_avg_auc = round(float(u_total_auc) / max(flag_count, 1), 4)
    return group_auc, u_avg_auc


def multi_auc(lis):
    preds = []
    labels = []
    uids = []
    online_scores = []
    res = []
    for l in lis:
        label = 1 if l[0] > 0 else 0
        pred_score = l[1]
        recID = l[2]
        uid = l[3]
        score = l[4]
        rank = l[5]

        labels.append(label)
        preds.append(pred_score)
        uids.append(recID)
        online_scores.append(score)

    auc_score = metrics.roc_auc_score(np.array(labels).astype('float'), np.array(preds).astype('float'))
    group_auc, u_avg_auc = cal_group_auc(labels, preds, uids)

    o_auc_score = metrics.roc_auc_score(np.array(labels).astype('float'), np.array(online_scores).astype('float'))
    o_group_auc, o_u_avg_auc = cal_group_auc(labels, online_scores, uids)
    return auc_score, group_auc, u_avg_auc, o_auc_score, o_group_auc, o_u_avg_auc


def load_yaml_file(yaml_file):
    f = open(yaml_file)
    ss = f.read()
    f.close()
    x = yaml.load(ss, Loader=yaml.FullLoader)
    return x


def save_yaml_file(x, yaml_file):
    fw = open(yaml_file, 'w')
    yaml.dump(x, fw)
    fw.close()


def load_conf(conf_file):
    conf = {}
    f = open(conf_file)
    for line in f:
        line = line.strip()
        if line == '':
            continue
        if line[0] == '#':
            continue
        lis = line.split('=')
        if len(lis) < 2:
            continue
        conf[lis[0]] = lis[1]
    f.close()
    return conf


def parse_line(line):
    lis = line.split('#', 1)[0].strip(' \n').split(' ')

    cvr_label = int(lis[0])

    add_info_list = line.split('#', 1)[1].strip(' \n').split('\t')
    cat_label = int(add_info_list[9])
    clk_label = int(add_info_list[8])
    ext_label = int(add_info_list[15])
    fea_index = []
    fea_value = []
    for i, x in enumerate(lis[1:]):
        y = x.split(':')
        ind = int(y[0])
        val = int(y[1])
        fea_index.append(ind)
        fea_value.append(val)

    add_info_list = [i.encode() for i in add_info_list]

    return fea_index, fea_value, cvr_label, cat_label, clk_label, ext_label,add_info_list


def WriteTFRecord(fs, outfile_name):
    writer = tf.io.TFRecordWriter(outfile_name)
    for line in fs:
        sids, fids, cvr_label, cat_label, clk_label,ext_label,add_info_list = parse_line(line)
        tfrecord_feature = {
            "cvr_label": tf.train.Feature(float_list=tf.train.FloatList(value=[cvr_label])),
            "cat_label": tf.train.Feature(float_list=tf.train.FloatList(value=[cat_label])),
            "clk_label": tf.train.Feature(float_list=tf.train.FloatList(value=[clk_label])),
            "ext_label": tf.train.Feature(float_list=tf.train.FloatList(value=[ext_label])),
            "sids": tf.train.Feature(int64_list=tf.train.Int64List(value=sids)),
            "fids": tf.train.Feature(int64_list=tf.train.Int64List(value=fids)),
            "add_infos": tf.train.Feature(bytes_list=tf.train.BytesList(value=add_info_list))
        }
        example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
        writer.write(example.SerializeToString())
    writer.close()


def ReadTFRecord(files, shuffle_size=1, batch_size=1, fetch_size=0, num_parallel=1):
    def parse_record(value):
        features = {
            "cvr_label": tf.io.FixedLenFeature([], tf.float32),
            "cat_label": tf.io.FixedLenFeature([], tf.float32),
            "clk_label": tf.io.FixedLenFeature([], tf.float32),
            "ext_label": tf.io.FixedLenFeature([], tf.float32),
            "sids": tf.io.VarLenFeature(tf.int64),
            "fids": tf.io.VarLenFeature(tf.int64),
            "add_infos": tf.io.VarLenFeature(tf.string)
        }
        parsed = tf.io.parse_single_example(value, features)

        return parsed

    def _is_day_directory(name):
        """检查是否为日期目录（8位数字格式，如 20260426）"""
        return len(name) == 8 and name.isdigit()

    def _expand_dir(path):
        """展开目录，按日期目录有序读取文件"""
        paths = tf.io.gfile.listdir(path)
        results = []
        
        # 分离日期目录和非日期目录/文件
        day_dirs = []
        other_items = []
        
        for p in paths:
            full_path = os.path.join(path, p)
            if tf.io.gfile.isdir(full_path):
                if _is_day_directory(p):
                    day_dirs.append((p, full_path))
                else:
                    other_items.append(full_path)
            else:
                # 文件直接添加（非日期目录下的文件）
                lower_name = p.lower()
                if lower_name.endswith(('.tfrecord', '.tfrecord.gz', '.record', '.record.gz', '.gz')):
                    results.append(full_path)
        
        # 按日期目录名排序（升序，确保日期有序）
        day_dirs.sort(key=lambda x: x[0])
        
        # 按日期顺序处理每个日期目录
        for day_name, day_path in day_dirs:
            day_files = tf.io.gfile.listdir(day_path)
            # 对日期目录内的文件排序
            day_files.sort()
            for f in day_files:
                full_path = os.path.join(day_path, f)
                if tf.io.gfile.isdir(full_path):
                    # 递归处理子目录
                    results.extend(_expand_dir(full_path))
                else:
                    lower_name = f.lower()
                    if lower_name.endswith(('.tfrecord', '.tfrecord.gz', '.record', '.record.gz', '.gz')):
                        results.append(full_path)
        
        # 处理非日期目录（递归）
        for other_path in other_items:
            results.extend(_expand_dir(other_path))
        
        return results

    def _expand_paths(path_list):
        expanded = []
        for path in path_list:
            if ',' in path:
                expanded.extend(_expand_paths(path.split(',')))
                continue
            if any(ch in path for ch in ['*', '?', '[']):
                matches = sorted(tf.io.gfile.glob(path))
            else:
                matches = [path]
            for matched in matches:
                if tf.io.gfile.isdir(matched):
                    expanded.extend(_expand_dir(matched))
                else:
                    expanded.append(matched)
        return expanded

    if isinstance(files, str):
        files = [files]
    elif isinstance(files, tuple):
        files = list(files)

    file_list = _expand_paths(files)
    if not file_list:
        raise ValueError('No TFRecord files found for path(s): %s' % files)
    
    # 打印文件列表以便调试
    print('Total %d TFRecord files found, reading in order:' % len(file_list))
    if len(file_list) > 0:
        print('First file:', file_list[0])
        if len(file_list) > 1:
            print('Last file:', file_list[-1])

    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, buffer_size=1024 * 1024 * 10, num_parallel_reads=num_parallel, compression_type="GZIP"),  # 10MB buffer
        cycle_length=num_parallel,  # 自动决定同时读几个文件
        num_parallel_calls=num_parallel,
        deterministic=False  # 允许乱序，速度更快
    )
    dataset = dataset.map(parse_record, num_parallel_calls=num_parallel)

    if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if fetch_size > 0:
        dataset = dataset.prefetch(fetch_size)
    return dataset

def ReadTFRecordV2(files, shuffle_size=1, batch_size=1, fetch_size=0, num_parallel=10):
    def parse_record(value):
        features = {
            "cvr_label": tf.io.FixedLenFeature([], tf.float32),
            "cat_label": tf.io.FixedLenFeature([], tf.float32),
            "clk_label": tf.io.FixedLenFeature([], tf.float32),
            "ext_label": tf.io.FixedLenFeature([], tf.float32),
            "sids": tf.io.VarLenFeature(tf.int64),
            "fids": tf.io.VarLenFeature(tf.int64),
            "add_infos": tf.io.VarLenFeature(tf.string)
        }
        parsed = tf.io.parse_single_example(value, features)
        return parsed

    def parse_record_batch(value):
        if isinstance(value, list):
            return map(parse_record, value)
        return parse_record(value)

    if ',' in files:
        files = files.split(',')
    if isinstance(files, list) or isinstance(files, tuple):
        pass
    else:
        files = [files]
    dataset = None
    tf_data_files = tf.data.Dataset.from_tensor_slices(files)
    dataset = tf_data_files.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(
            x,
            buffer_size=100000000,
            num_parallel_reads=4,
            compression_type="GZIP",
        ),
        num_parallel_calls=num_parallel,
        cycle_length=num_parallel,
        block_length=32
    )
    dataset = dataset.map(parse_record, num_parallel_calls=num_parallel)
    if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if fetch_size > 0:
        dataset = dataset.prefetch(fetch_size)
    return dataset

if __name__ == '__main__':
    WriteTFRecord(sys.stdin, sys.argv[1])
