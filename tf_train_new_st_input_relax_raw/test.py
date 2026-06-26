import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
import datetime
import model_conf
from model import Model
from datetime import timedelta
from tensorflow.keras import regularizers
import time

class Learner:
    def __init__(self):
        self.model = None

    def set_training_mode(self, enable_training):
        self.model.training = enable_training

    def get_files(self, path, start, end):
        d = datetime.datetime.strptime(start, "%Y%m%d")
        end = datetime.datetime.strptime(end, "%Y%m%d")
        files = []
        while d <= end:
            files += tf.io.gfile.glob("{path}/{d}/part*".format(path=path,d=d.strftime("%Y%m%d")))
            d += timedelta(1)
        files = sorted(files)
        return files    

    def train(self, train_data, model_path=None, data_path=None):
        if self.model is None:
            self.model = Model(training=True)

        batch_size = model_conf.batch_size
        epoch_num = model_conf.epoch_num

        model = self.model

        #load ckpt
        ckpt_path = self.get_model_checkpoint_from_file(model_conf.done_file_path)
        #ckpt_path="model/checkpoints/20260430_0/"
        if ckpt_path is not None:
            print("load model from checkpoint:", ckpt_path)
            ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

            first_batch = next(iter(train_data))
            _ = model([first_batch['sids'], first_batch['fids']])

            dummy_grad = [tf.zeros_like(v) for v in model.trainable_variables]
            model.optimizer.apply_gradients(zip(dummy_grad, model.trainable_variables))

            #ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).expect_partial()
            ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).assert_consumed()
            print("Restored optimizer step: ", model.optimizer.iterations.numpy())
            print("load checkpoint path: ", ckpt_path)

        buy_weight = 1.0
        cat_weight = 1.0
        click_weight = 1.0
        ext_weight = 1.0
        print('testing...')
        self.set_training_mode(False)
        self.test(train_data)
        self.set_training_mode(True)

    def test(self, test_data, model_path=''):

        res_buy = []
        res_cat = []
        res_click = []
        res_ext = []
        loss_buy_sum = 0.0
        loss_cat_sum = 0.0
        loss_click_sum = 0.0
        loss_ext_sum = 0.0
        pos_buy = 0
        pos_cat = 0
        pos_click = 0
        pos_ext = 0

        buy_weight = 0.8
        click_weight = 0.2

        low_score_num = 0

        for step, feat in enumerate(test_data):
            pred_buy, pred_cat, pred_click, pred_ext,gate1, gate2, gate3 ,gate4 = self.model([feat['sids'], feat['fids']])

            loss_buy = self.model.loss(tf.expand_dims(feat['cvr_label'], 1), pred_buy)
            loss_cat = self.model.loss(tf.expand_dims(feat['cat_label'], 1), pred_cat)
            loss_click = self.model.loss(tf.expand_dims(feat['clk_label'], 1), pred_click)
            loss_ext = self.model.loss(tf.expand_dims(feat['ext_label'], 1), pred_ext)

            loss_buy_sum += tf.reduce_sum(loss_buy, 0)
            loss_cat_sum += tf.reduce_sum(loss_cat, 0)
            loss_click_sum += tf.reduce_sum(loss_click, 0)
            loss_ext_sum += tf.reduce_sum(loss_ext, 0)

            #pred_buy = pred_buy * pred_click
            #pred_cat = pred_cat * pred_click
            pred_buy = tf.squeeze(pred_buy, 1).numpy().tolist()
            pred_cat = tf.squeeze(pred_cat, 1).numpy().tolist()
            pred_click = tf.squeeze(pred_click, 1).numpy().tolist()
            pred_ext = tf.squeeze(pred_ext, 1).numpy().tolist()

            buy_label = feat['cvr_label'].numpy().reshape(-1).tolist()
            cat_label = feat['cat_label'].numpy().reshape(-1).tolist()
            click_label = feat['clk_label'].numpy().reshape(-1).tolist()
            ext_label = feat['ext_label'].numpy().reshape(-1).tolist()

            recID = [i.decode() for i in feat['add_infos'].values.numpy()[6:][::24]]
            uid = [i.decode() for i in feat['add_infos'].values.numpy()[5:][::24]]
            score = [i.decode() for i in feat['add_infos'].values.numpy()[4:][::24]]
            rank = [i.decode() for i in feat['add_infos'].values.numpy()[3:][::24]]

            pos_buy += sum(buy_label)
            pos_cat += sum(cat_label)
            pos_click += sum(click_label)
            pos_ext += sum(ext_label)

            # add_info = feat['add_infos'].numpy().reshape(-1).tolist()

            res_buy.extend(zip(buy_label, pred_buy, recID, uid, score, rank))
            res_cat.extend(zip(cat_label, pred_cat, recID, uid, score, rank))
            res_click.extend(zip(click_label, pred_click, recID, uid, score, rank))
            res_ext.extend(zip(ext_label, pred_ext, recID, uid, score, rank))
            low_score_num += len(list((filter(lambda x: x < 0.01, pred_buy))))

            if step % 10000 == 0:
                print(datetime.datetime.now(),
                      "step :%d buy loss:%04f, pos: %d, cat loss:%04f, pos cat: %d, click loss:%04f, pos2: %d ,ext loss:%04f, pos ext: %d" % (
                          step, loss_buy_sum / len(res_buy), pos_buy, loss_cat_sum / len(res_cat), pos_cat,
                          loss_click_sum / len(res_click), pos_click, loss_ext_sum / len(res_ext), pos_ext))

        print(model_path, "low score rate:%f" % (low_score_num / len(res_buy)))

        auc_score, group_auc, u_avg_auc, o_auc_score, o_group_auc, o_u_avg_auc = ut.multi_auc(res_buy)
        print(model_path, "test_buy auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score, group_auc, u_avg_auc, len(res_buy), loss_buy_sum / len(res_buy), pos_buy))
        print(model_path, "online_buy auc:%f gauc:%f uauc:%f " % (o_auc_score, o_group_auc, o_u_avg_auc))

        auc_score_cat, group_auc_cat, u_avg_auc_cat, o_auc_score_cat, o_group_auc_cat, o_u_avg_auc_cat = ut.multi_auc(
            res_cat)
        print(model_path, "test_cat auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score_cat, group_auc_cat, u_avg_auc_cat, len(res_cat), loss_cat_sum / len(res_cat), pos_cat))
        print(model_path, "online_cat auc:%f gauc:%f uauc:%f " % (o_auc_score_cat, o_group_auc_cat, o_u_avg_auc_cat))

        auc_score_click, group_auc_click, u_avg_auc_click, o_auc_score_click, o_group_auc_click, o_u_avg_auc_click = ut.multi_auc(
            res_click)
        print(model_path, "test_click auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score_click, group_auc_click, u_avg_auc_click, len(res_click), loss_click_sum / len(res_click),
            pos_click))
        print(model_path,
              "online_click auc:%f gauc:%f uauc:%f " % (o_auc_score_click, o_group_auc_click, o_u_avg_auc_click))

        auc_score_ext, group_auc_ext, u_avg_auc_ext, o_auc_score_ext, o_group_auc_ext, o_u_avg_auc_ext = ut.multi_auc(
            res_ext)
        print(model_path, "test_ext auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score_ext, group_auc_ext, u_avg_auc_ext, len(res_ext), loss_ext_sum / len(res_ext),
            pos_ext))
        print(model_path,
              "online_ext auc:%f gauc:%f uauc:%f " % (o_auc_score_ext, o_group_auc_ext, o_u_avg_auc_ext))

        pass

    def dump_serving_model(self, end_day, epo):
        if self.model is None:
            return
        self.model.training = False

        train_model = self.model

        fid_keys, fid_values = train_model.fid_table.export()
        fid_keys_ads, fid_values_ads = train_model.fid_table_din_ads.export()

        fid_keys = tf.reshape(fid_keys, [-1])
        fid_values = tf.reshape(fid_values, [-1])

        fid_keys_ads = tf.reshape(fid_keys_ads, [-1])
        fid_values_ads = tf.reshape(fid_values_ads, [-1])

        serve_model = Model(
            training=False,
            pred=True,
            fid_kv=(fid_keys, fid_values),
            fid_ads_kv=(fid_keys_ads, fid_values_ads)
        )
        serve_model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=['mae'])

        serve_model.is_save_model = True
        dummy_sids = tf.constant([[0] * model_conf.padding_size], tf.uint32)
        dummy_fids = tf.constant([[0] * model_conf.padding_size], tf.uint64)
        _ = serve_model([dummy_sids, dummy_fids])

        serve_model._set_inputs([
            tf.keras.Input(shape=(model_conf.padding_size,), dtype=tf.dtypes.uint32),
            tf.keras.Input(shape=(model_conf.padding_size,), dtype=tf.dtypes.uint64),
        ])

        #serve_model.load_weights(model_conf.model_path)
        serve_model.set_weights(train_model.get_weights())

        serve_model.save(
            'serving_model_%s/%s00' % (epo, end_day),
            save_format='tf'
        )

    def predict(self, X, model_path=None):
        if self.model is None:
            self.init()
            if model_path is not None:
                self.model._set_inputs(X)
                self.model.load_weights(model_path)
        self.model.pred = True
        self.count_parameters(verbose=True)
        pred = self.model(X)
        return tf.squeeze(pred, 1).numpy().tolist()

    def count_parameters(self, verbose=False):
        total_params = 0
        if self.model is None:
            print("Model not initialized")
            return
        for i,weight in enumerate(self.model.trainable_weights):
            shape = weight.shape.as_list()
            num_params = np.prod(shape)
            total_params += num_params
            if verbose:
                print("Layer "+str(i)+":"+str(weight.name)+" | Shape:"+str(shape)+" | params:"+str(num_params))
                #print(f"Layer {i} :{weight.name} | Shape : {shape} |Params : {num_params} ")
        #print(f"\nTotal trainable parameters : {total_params}")
        print("Total trainable parameters:"+str(total_params))
        return total_params

    def batch_test_local(self, data_set, model_path, batch_size=1024):
        # init models
        res = []
        model_list = []
        lis = model_path.split(',')
        print('lis={}'.format(lis))
        for x in lis:
            model_list.append(x)
            md = copy.deepcopy(self)
            md.init()
            md.model._set_inputs([data_set.__iter__().next()['fea_ids'], data_set.__iter__().next()['fea_vals']])
            md.model.load_weights(x)
            md.test(data_set, x)

    def batch_test(self, infs, model_path, batch_size=1024, ofs=None, is_test=True, output_infos=False):
        indices = []
        fea_index = []
        fea_value = []
        labels = []
        infos = []
        mx = -1
        NR = 0
        pos_num = 0
        res = []
        model_list = []
        lis = model_path.split(',')
        for x in lis:
            model_list.append([copy.deepcopy(self), x])
            res.append([])
        for line in infs:
            NR += 1
            lis = line.split('#', 1)[0].strip(' \n').split(' ')
            info = line.split('#', 1)[1].strip(' \n')
            infos.append(info)
            label = int(lis[0])
            if label > 0:
                pos_num += 1
            labels.append(label)
            for i, x in enumerate(lis[1:]):
                y = x.split(':')
                ind = int(y[0])
                val = float(y[1])
                indices.append([NR - 1, i])
                fea_index.append(ind)
                fea_value.append(val)
                if i > mx:
                    mx = i
            if NR == batch_size:
                # X_ind=tf.sparse.to_dense(tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx+1]))
                # X_val=tf.sparse.to_dense(tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx+1]))
                X_ind = tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx + 1])
                X_val = tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx + 1])
                for i, model in enumerate(model_list):
                    pred = model[0].predict([X_ind, X_val], model[1])
                    res[i].extend(zip(labels, pred, infos))
                indices = []
                fea_index = []
                fea_value = []
                infos = []
                labels = []
                NR = 0

        if NR > 0:
            X_ind = tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx + 1])
            X_val = tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx + 1])
            for i, model in enumerate(model_list):
                pred = model[0].predict([X_ind, X_val], model[1])
                res[i].extend(zip(labels, pred, infos))
        if is_test:
            for i, model in enumerate(model_list):
                print("%s test auc:%f size:%d" % (model[1], ut.auc(res[i]), len(res[i])))

        print(ofs)
        if ofs is not None:
            model_id = -1
            for r in res:
                model_id += 1
                idx = 0
                for x in r:
                    if output_infos:
                        print('\t'.join(['tfmodel_' + str(model_id), str(x[0]), str(x[1]), x[2]]), file=ofs)  # lmy
                        idx += 1  # lmy
                    else:
                        print(x[1], file=ofs)

    def get_model_checkpoint_from_file(self, done_file_path='model.done'):
        if not os.path.exists(done_file_path):
            print("model.done not exit in patch: ",done_file_path)
            return None

        with open(done_file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("model.done is null ")
                return None

            # read last line
            last_line = lines[-1].strip()
            if not last_line:
                print("model.done last line is null")
                return None

            # file format: checkpoint_day\tcheckpoint_path
            parts = last_line.split('\t')
            if len(parts) >= 2:
                ckpt_day = parts[0]
                ckpt_path = parts[1]
                print("load model checkpoint_path=%s, checkpoint_day=%s",ckpt_path, ckpt_day)
                return ckpt_path
            else:
                print("model.done last line format error")
                return None

if __name__ == "__main__":
    # init args and model
    parse = argparse.ArgumentParser(description='get input args')
    parse.add_argument('-data', type=str, help='input data files')
    parse.add_argument('-start_day', type=str, help='train start day')
    parse.add_argument('-end_day', type=str, help='train end day')

    args = parse.parse_args()
    solver = Learner()

    # set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = model_conf.gpu_id
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # start training or testing
    if model_conf.train_mode == 'train':
        batch_size = model_conf.batch_size
        shuffle_size = batch_size * 10

        #read data
        print('start read data')
        start_time = time.time()
        files = solver.get_files(args.data, args.start_day, args.end_day)
        print("files: ", files)
        ds = ut.ReadTFRecordV2(files, shuffle_size=shuffle_size, batch_size=batch_size, fetch_size=10, num_parallel=10)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        end_time = time.time()
        using_time = end_time - start_time
        print('end read data, using_time_reading_data: ', using_time)

        #start training
        print('start training')
        solver.train(ds, data_path=args.data)
        end_time2 = time.time()
        using_time2 = end_time2 - end_time
        print('end training, using_time_training: ', using_time2)
    elif model_conf.train_mode == 'test_xxx':
        solver.init()
        info = {}
        ds = ut.svmlight2dataset(sys.stdin, info)
        print("total ins num:", info['all_num'])
        print("postive ins num:", info['pos_num'])
        ds = ds.batch(1024)
        solver.model._set_inputs([ds.__iter__().next()['fea_ids'], ds.__iter__().next()['fea_vals']])
        solver.model.load_weights(model_conf.model_path)
        solver.test(ds)
        pass
    elif model_conf.train_mode == 'test':
        if args.data:
            print('start read data, data=', args.data)
            start_time = time.time()
            ds = ut.ReadTFRecord(args.data, batch_size=1024, fetch_size=1)
            end_time = time.time()
            using_time = end_time - start_time
            print('end read data, using_time_reading_data: ', using_time)

            print('start training')
            solver.batch_test_local(ds, model_conf.model_path)
            end_time2 = time.time()
            using_time2 = end_time2 - end_time
            print('end training, using_time_training: ', using_time2)
        else:
            solver.batch_test(sys.stdin, model_conf.model_path)
    elif model_conf.train_mode== 'predict':
        if args.dataOut:
            f = open(args.dataOut, 'w')
            solver.batch_test(sys.stdin, model_conf.model_path, ofs=f, is_test=False)
            f.close()
        else:
            solver.batch_test(sys.stdin, model_conf.model_path, ofs=sys.stdout, is_test=False)
    elif model_conf.train_mode == 'predict_info':
        solver.batch_test(sys.stdin, model_conf.model_path, ofs=sys.stdout, is_test=False, output_infos=True,)
    elif model_conf.train_mode == 'dump_serving':
        solver.dump_serving_model(end_day=args.end_day)
    else:
        pass
