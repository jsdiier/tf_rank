import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
import datetime
from datetime import timedelta
import model_conf
import wandb
from model import Model
from tensorflow.keras import regularizers
import time

class Learner:
    def __init__(self):
        self.model = None

    def set_training_mode(self, enable_training, is_save_model):
        self.model.training = enable_training
        self.model.is_save_model = is_save_model

    def get_files(self, path, start, end):
        d = datetime.datetime.strptime(start, "%Y%m%d")
        end = datetime.datetime.strptime(end, "%Y%m%d")
        files = []
        while d <= end:
            files += tf.io.gfile.glob("{path}/{d}/part*".format(path=path,d=d.strftime("%Y%m%d")))
            d += timedelta(1)
        files = sorted(files)
        return files

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, feat, buy_weight, cat_weight, click_weight, ext_weight):
        model = self.model
        with tf.GradientTape() as tape:
            pred_buy, pred_cat, pred_click, pred_ext, gate1, gate2, gate3 ,gate4 = model([feat['sids'], feat['fids']], training=True)

            loss_buy = model.loss(tf.expand_dims(feat['cvr_label'], 1), pred_buy)
            loss_cat = model.loss(tf.expand_dims(feat['cat_label'], 1), pred_cat)
            loss_click = model.loss(tf.expand_dims(feat['clk_label'], 1), pred_click)
            loss_ext = model.loss(tf.expand_dims(feat['ext_label'], 1), pred_ext)

            final_loss = (
                loss_buy * buy_weight
                + loss_cat * cat_weight
                + loss_click * click_weight
                + loss_ext * ext_weight
            )

            gradients = tape.gradient(final_loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss_buy, loss_cat, loss_click, loss_ext, final_loss

    def train(self, train_data, start_day, end_day, model_path=None, data_path=None):
        if self.model is None:
            self.model = Model(training=True)

        batch_size = model_conf.batch_size
        epoch_num = model_conf.epoch_num

        model = self.model

        train_writer = None
        tensorboard_dir = "./log/tensorboard_data_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if tensorboard_dir:
            train_writer = tf.summary.create_file_writer(
                os.path.join(tensorboard_dir, "train"), max_queue=1000
            )
            model.set_summary_writer(train_writer, histogram_freq=100)
            print('TensorBoard enabled:', tensorboard_dir)

        #load ckpt
        ckpt_path = self.get_model_checkpoint_from_file(model_conf.done_file_path)
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

        wandb.init(
            project="tf_rank",
            name=os.environ.get("JOB_NAME", "tf_train_default"),
            config={
                "batch_size":   model_conf.batch_size,
                "lr":           model_conf.learning_rate,
                "fm_emb_size":  model_conf.fm_emb_size,
                "din_emb_size": model_conf.din_emb_size,
                "num_buckets":  model_conf.num_buckets,
                "feature_size": model_conf.feature_size,
                "buy_weight":   buy_weight,
                "cat_weight":   cat_weight,
                "click_weight": click_weight,
                "ext_weight":   ext_weight,
            },
        )
        print('training...')
        for epo in range(epoch_num):
            pos = 0
            pos_cat = 0
            pos_click = 0
            pos_ext = 0
            cnt = 0
            print("train status:", self.model.training)
            print("is_save_model:", self.model.is_save_model)
            for step, feat in enumerate(train_data):
                cnt += feat['cvr_label'].shape[0]
                pos += sum(feat['cvr_label'].numpy())
                pos_cat += sum(feat['cat_label'].numpy())
                pos_click += sum(feat['clk_label'].numpy())
                pos_ext += sum(feat['ext_label'].numpy())

                loss_buy, loss_cat, loss_click, loss_ext, final_loss = self.train_step(feat, buy_weight, cat_weight, click_weight, ext_weight)

                if step % 100 == 0:
                    if train_writer is not None:
                        global_step = model.optimizer.iterations.numpy()
                        with train_writer.as_default():
                            tf.summary.scalar('loss_buy', tf.reduce_mean(loss_buy), step=global_step)
                            tf.summary.scalar('loss_cat', tf.reduce_mean(loss_cat), step=global_step)
                            tf.summary.scalar('loss_click', tf.reduce_mean(loss_click), step=global_step)
                            tf.summary.scalar('loss_ext', tf.reduce_mean(loss_ext), step=global_step)
                            tf.summary.scalar('loss/total', tf.reduce_mean(final_loss), step=global_step)

                            tf.summary.scalar('data/pos_rate_buy', pos / max(cnt, 1), step=global_step)
                            tf.summary.scalar('data/pos_rate_click', pos_click / max(cnt, 1), step=global_step)

                    print(datetime.datetime.now(),
                            "step No.%08d\t buy loss:%04f, pos: %d, cnt: %d,  cat loss:%04f, cat pos: %d,  click loss:%04f, click pos: %d,  ext loss:%04f, ext pos: %d" % (
                            step, tf.reduce_mean(loss_buy), pos, cnt, tf.reduce_mean(loss_cat), pos_cat,
                            tf.reduce_mean(loss_click), pos_click, tf.reduce_mean(loss_ext), pos_ext))

                    wandb.log({
                        "loss/buy":   float(tf.reduce_mean(loss_buy)),
                        "loss/cat":   float(tf.reduce_mean(loss_cat)),
                        "loss/click": float(tf.reduce_mean(loss_click)),
                        "loss/ext":   float(tf.reduce_mean(loss_ext)),
                        "loss/total": float(tf.reduce_mean(final_loss)),
                        "data/pos_rate_buy":   pos / max(cnt, 1),
                        "data/pos_rate_cat":   pos_cat / max(cnt, 1),
                        "data/pos_rate_click": pos_click / max(cnt, 1),
                        "data/pos_rate_ext":   pos_ext / max(cnt, 1),
                        "progress/step": step,
                        "progress/cnt":  cnt,
                    }, step=int(model.optimizer.iterations.numpy()))
            print(datetime.datetime.now(), "epo no:%d finish" % (epo))


            if train_writer is not None:
                train_writer.flush()

            #dump ckpt
            if epo >= 0:
                save_dir = "%s/checkpoints/%s_%s/" % (model_conf.local_model_dir, end_day, epo)
                export_dir = save_dir + "tfmodel"
                ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
                ckpt.save(export_dir)

                # Write checkpoint info to done file
                save_day_time = end_day
                done_dir = os.path.dirname(model_conf.done_file_path)
                if done_dir and not os.path.exists(done_dir):
                    try:
                        os.makedirs(done_dir)
                    except Exception as e:
                        print("Warning: Failed to create done_file directory {done_dir}:", e)
                try:
                    with open(model_conf.done_file_path, 'a') as f:
                        f.write(save_day_time + "\t" + save_dir + "\n")
                except Exception as e:
                    print("Warning: Failed to write done file {model_conf.done_file_path}:", e)

            self.set_training_mode(False, True)
            if epo >= 0:
                self.dump_serving_model(end_day, epo)
            self.set_training_mode(True, False)

    def dump_serving_model(self, end_day, epo):
        if self.model is None:
            return

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

        serve_model.training = False
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
        solver.train(ds, data_path=args.data, start_day=args.start_day, end_day=args.end_day)
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