import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import model_conf
from tensorflow.python.framework import sparse_tensor

class Model(tf.keras.Model):
    def __init__(self, training=False, pred=False, fid_kv=None, fid_ads_kv=None):
        super(Model, self).__init__()

        self.is_save_model = False
        self.training = training
        self.pred = pred
        self.dropout_dim = []
        self.summary_writer = None

        self.lhuc_layers_cache = {}
        self.attention_layers_cache = {}
        self.bottom_layers_cache = {}
        self.task_layers_cache = {}
        self.ads_layers_cache = {}

        self.loss = tf.keras.losses.binary_crossentropy
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(model_conf.learning_rate, decay_steps=1000000,
                                                                          decay_rate=1, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-07, amsgrad=False, name='Adam')

        #embedding table
        self.emb_fm = tf.keras.layers.Embedding(
            model_conf.feature_size,
            model_conf.lr_emb_size + model_conf.fm_emb_size,
            embeddings_regularizer=regularizers.l2(model_conf.l2_reg))
        #self.emb_lr = tf.keras.layers.Embedding(
        #    model_conf.feature_size,
        #    model_conf.lr_emb_size,
        #    embeddings_regularizer=regularizers.l2(model_conf.l2_reg))
        self.emb_din_ads = tf.keras.layers.Embedding(
            model_conf.ads_fea_size,
            model_conf.din_emb_size,
            embeddings_regularizer=regularizers.l2(model_conf.l2_reg))

        self.slot_id_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(model_conf.all_slot_ids, dtype=tf.int32),
                values=tf.range(len(model_conf.all_slot_ids), dtype=tf.int32)
            ),
            default_value=-1
        )

        if fid_kv is not None:
            self.fid_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=fid_kv[0],
                    values=fid_kv[1]
                ),
                default_value=-1
            )
        else:
            self.fid_table = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.int64,
                default_value=-1,
                empty_key=0,
                deleted_key=-1,
                initial_num_buckets=model_conf.num_buckets
            )
        self.counter = tf.Variable(1, dtype=tf.int64, trainable=False)

        self.slot_id_table_din_ads = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(model_conf.slot_id_v2, dtype=tf.int32),
                values=tf.range(len(model_conf.slot_id_v2), dtype=tf.int32)
            ),
            default_value=-1
        )
        if fid_ads_kv is not None:
            self.fid_table_din_ads = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=fid_ads_kv[0],
                    values=fid_ads_kv[1]
                ),
                default_value=-1
            )
        else:
            self.fid_table_din_ads = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.int64,
                default_value=-1,
                empty_key=0,
                deleted_key=-1,
                initial_num_buckets=model_conf.ads_num_buckets
            )
        self.counter_din_ads = tf.Variable(1, dtype=tf.int64, trainable=False)

        #self.bn = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=1e-3,center=True,scale=False)

        self.query_dense = tf.keras.layers.Dense(model_conf.fm_emb_size, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(model_conf.l2_reg))

        self.dense_concat = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat1 = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat2 = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat3 = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(model_conf.l2_reg))

    def set_summary_writer(self, writer, histogram_freq=100):
        self.summary_writer = writer
        self.histogram_freq = histogram_freq

    def _write_histograms(self, step, gradients=None):
        with self.summary_writer.as_default():
            grad_map = {}
            if gradients is not None:
                for var, grad in zip(self.trainable_weights, gradients):
                    grad_map[var.name] = grad

            for var in self.trainable_weights:
                name_lower = var.name.lower()
                if any(k in name_lower for k in ["dense", "gate", "emb"]):
                    safe_name = var.name.replace(':', '_')
                    tf.summary.histogram(safe_name, var, step=step)
                    grad = grad_map.get(var.name)
                    if grad is not None:
                        tf.summary.histogram(safe_name + "_grad", grad, step=step)

    def transform(self, sids, fids):
        if self.is_save_model:
            sid_list = tf.cast(sids, tf.dtypes.int32)
            fid_list = tf.cast(fids, tf.dtypes.int64)
        else:
            sid_list = tf.sparse.to_dense(sids)
            fid_list = tf.sparse.to_dense(fids)
            sid_list = tf.cast(sid_list, tf.dtypes.int32)
        return sid_list, fid_list

    def lhuc_net(self, nn_input, nn_dims, name, lhuc_input, lhuc_basic_dims, dropout_dim, is_train, use_bn=False):
        cur_layer = nn_input
        nn_input_dim = nn_input.shape[-1]
        lhuc_dims = [nn_input_dim] + nn_dims[:-1]

        layer_key = name
        if layer_key not in self.lhuc_layers_cache:
            layers_dict = {}

            if use_bn:
                bn_layer = tf.keras.layers.BatchNormalization(name=name+'_bn')
                layers_dict['bn_layer'] = bn_layer

            for i in range(len(nn_dims)):
                lhuc_ds = lhuc_basic_dims + [lhuc_dims[i]]
                lhuc_layers = []

                for j in range(len(lhuc_ds)):
                    ldim = lhuc_ds[j]
                    lhuc_layer = tf.keras.layers.Dense(
                        units=ldim,
                        activation=tf.nn.relu,
                        name=name+'_'+ str(i) + '_lhuc_' + str(j),
                        kernel_regularizer=regularizers.l2(0.001)
                    )
                    lhuc_layers.append(lhuc_layer)

                dense_layer = tf.keras.layers.Dense(
                    units=nn_dims[i],
                    activation=tf.nn.relu,
                    name=name+'_'+str(i)+'_dense',
                    kernel_regularizer=regularizers.l2(0.001)
                )

                dropout_layer = None
                if i + 1 < len(dropout_dim) and dropout_dim[i + 1] > 1e-6:
                    dropout_layer = tf.keras.layers.Dropout(
                        rate=dropout_dim[i + 1],
                        name=name+'_'+str(i)+'_dropout'
                    )

                ln_layer = tf.keras.layers.LayerNormalization(name=name+'_'+str(i)+'_ln')
                layers_dict["lhuc_layers_"+str(i)] = lhuc_layers
                layers_dict["dense_layer_"+str(i)] = dense_layer
                layers_dict["dropout_layer_"+str(i)] = dropout_layer
                layers_dict["ln_layer_"+str(i)] = ln_layer

            self.lhuc_layers_cache[layer_key] = layers_dict

        cached_layers = self.lhuc_layers_cache[layer_key]
        if use_bn and 'bn_layer' in cached_layers:
            cur_layer = cached_layers['bn_layer'](cur_layer, training=is_train)

        for i in range(len(nn_dims)):
            lhuc_layers = cached_layers["lhuc_layers_"+str(i)]
            dense_layer = cached_layers["dense_layer_"+str(i)]
            dropout_layer = cached_layers["dropout_layer_"+str(i)]
            ln_layer = cached_layers["ln_layer_"+str(i)]
            lhuc_output = lhuc_input
            for j, lhuc_layer in enumerate(lhuc_layers):
                lhuc_output = lhuc_layer(lhuc_output)

            lhuc_output = 1.0 + 0.5 * tf.nn.tanh(0.2 * lhuc_output)
            cur_layer = cur_layer * lhuc_output
            cur_layer = dense_layer(cur_layer)
            cur_layer = ln_layer(cur_layer)

            if dropout_layer is not None:
                cur_layer = dropout_layer(cur_layer, training=is_train)

        return cur_layer

    def fid_lookup_or_insert(self, valid_fids, table_type):
        flat_ids = tf.reshape(valid_fids, [-1])
        unique_ids, idx_map = tf.unique(flat_ids)

        empty_key = 0
        valid_mask = tf.not_equal(unique_ids, empty_key)
        unique_ids = tf.boolean_mask(unique_ids, valid_mask)

        if table_type == 'din_ads_table':
            unique_mapped = self.fid_table_din_ads.lookup(unique_ids)
        else:
            unique_mapped = self.fid_table.lookup(unique_ids)

        mask = tf.equal(unique_mapped, -1)

        def insert_and_update():
            miss_indices = tf.where(mask)
            miss_ids = tf.gather_nd(unique_ids, miss_indices)
            num_new = tf.shape(miss_ids)[0]

            if table_type == 'din_ads_table':
                current_count = self.counter_din_ads.read_value()
                new_values = tf.range(current_count, current_count + tf.cast(num_new, tf.int64))
                self.counter_din_ads.assign_add(tf.cast(num_new, tf.int64))

                self.fid_table_din_ads.insert(miss_ids, new_values)
            else:
                current_count = self.counter.read_value()
                new_values = tf.range(current_count, current_count + tf.cast(num_new, tf.int64))
                self.counter.assign_add(tf.cast(num_new, tf.int64))

                self.fid_table.insert(miss_ids, new_values)

            return tf.tensor_scatter_nd_update(
                unique_mapped, miss_indices, new_values
            )

        final_unique_mapped = tf.cond(
            tf.reduce_any(mask),
            true_fn=insert_and_update,
            false_fn=lambda: unique_mapped
        )

        result = tf.gather(final_unique_mapped, idx_map)
        return tf.reshape(result, tf.shape(valid_fids))

    def fid_lookup(self, valid_fids, table_type):
        flat_ids = tf.reshape(valid_fids, [-1])

        if table_type == 'din_ads_table':
            mapped_ids = self.fid_table_din_ads.lookup(flat_ids)
        else:
            mapped_ids = self.fid_table.lookup(flat_ids)

        result = tf.reshape(mapped_ids, tf.shape(valid_fids))
        return result

    def process_and_pool_fused(self, sid_list, fid_list, table_type='emb_table'):
        batch_size = tf.shape(sid_list)[0]
        n = tf.shape(sid_list)[1]
        sid_list_flat = tf.reshape(sid_list, [-1])
        fid_list_flat = tf.reshape(fid_list, [-1])

        if table_type == 'din_ads_table':
            num_segments = len(model_conf.slot_id_v2)
            mapped_indices = self.slot_id_table_din_ads.lookup(sid_list_flat)
        else:
            num_segments = len(model_conf.all_slot_ids)
            mapped_indices = self.slot_id_table.lookup(sid_list_flat)

        mask = tf.not_equal(mapped_indices, -1)
        valid_fids = tf.boolean_mask(fid_list_flat, mask)
        valid_indices = tf.boolean_mask(mapped_indices, mask)

        batch_ids = tf.repeat(tf.range(batch_size), n)
        valid_batch_ids = tf.boolean_mask(batch_ids, mask)
        combined_indices = valid_batch_ids * num_segments + valid_indices

        if self.is_save_model or self.pred:
            new_fid_list = self.fid_lookup(valid_fids, table_type)
        else:
            new_fid_list = self.fid_lookup_or_insert(valid_fids, table_type)

        if table_type == 'din_ads_table':
            vocab_size = model_conf.ads_fea_size
        else:
            vocab_size = model_conf.feature_size

        valid_new_fids = (new_fid_list > 0) & (new_fid_list < vocab_size)
        new_fid_list = tf.boolean_mask(new_fid_list, valid_new_fids)
        combined_indices = tf.boolean_mask(combined_indices, valid_new_fids)

        if table_type == 'din_ads_table':
            embeds = self.emb_din_ads(new_fid_list)
        else:
            embeds = self.emb_fm(new_fid_list)
            #lr_embedding = self.emb_lr(new_fid_list)
            #embeds = tf.concat([lr_embedding, fm_embedding], axis=1)

        pooled_flat = tf.math.unsorted_segment_mean(
            data=embeds,
            segment_ids=combined_indices,
            num_segments=batch_size * num_segments
        )

        embedding_dim = tf.shape(embeds)[1]
        pooled_output = tf.reshape(pooled_flat,[batch_size, num_segments, embedding_dim])

        #slot mask
        ones = tf.ones_like(combined_indices, dtype=tf.float32)
        counts_flat = tf.math.unsorted_segment_mean(
            data=ones,
            segment_ids=combined_indices,
            num_segments=batch_size * num_segments
        )
        counts = tf.reshape(counts_flat, [batch_size, num_segments])
        slot_mask = tf.cast(tf.greater(counts, 0), dtype=tf.float32)

        return pooled_output, slot_mask

    def ads_seq_cross_layer(self, name, nn_inputs, ads_emb, ads_hidden_dim=64, ads_output_dim=1):
        #ads_input_dim = nn_inputs.get_shape().as_list()[-1]
        ads_input_dim = tf.shape(nn_inputs)[-1]

        layer_key = name

        if layer_key not in self.ads_layers_cache:
            ads_weight_dims = [ads_hidden_dim, ads_input_dim * ads_output_dim]
            ads_weight_acts = [tf.nn.relu, None]
            ads_weight_layers = []

            for i in range(len(ads_weight_dims)):
                pdim = ads_weight_dims[i]
                pact = ads_weight_acts[i]
                layer = tf.keras.layers.Dense(
                    units=pdim,
                    activation=pact,
                    name=name+'_ads_weight_'+str(i)
                )
                ads_weight_layers.append(layer)

            ads_bias_dims = [ads_hidden_dim, ads_output_dim]
            ads_bias_layers = []
            for i in range(len(ads_bias_dims)):
                pdim = ads_bias_dims[i]
                layer = tf.keras.layers.Dense(
                    units=pdim,
                    activation=tf.nn.relu,
                    name=name+'_ads_bias_'+str(i)
                )
                ads_bias_layers.append(layer)

            self.ads_layers_cache[layer_key] = {
                'weight_layers': ads_weight_layers,
                'bias_layers': ads_bias_layers
            }

        cached_layers = self.ads_layers_cache[layer_key]
        ads_weight_layers = cached_layers['weight_layers']
        ads_bias_layers = cached_layers['bias_layers']

        ads_weight = ads_emb
        for i, layer in enumerate(ads_weight_layers):
            ads_weight = layer(ads_weight)
        ads_weight = tf.reshape(ads_weight, [-1, ads_input_dim, ads_output_dim])

        ads_bias = ads_emb
        for i, layer in enumerate(ads_bias_layers):
            ads_bias = layer(ads_bias)
        ads_bias = tf.expand_dims(ads_bias, 1)

        output = tf.matmul(nn_inputs, ads_weight)
        output = tf.add(output, ads_bias)

        return output

    def attention_din_ads(self, query, key, mask, ads_emb, name, att_hidden_units):
        ads_key = self.ads_seq_cross_layer(name=name+'_ads_layer',
                    nn_inputs=key,
                    ads_emb=ads_emb,
                    ads_hidden_dim=64,
                    ads_output_dim=tf.shape(key)[-1])

        query_dim = tf.shape(query)[-1]
        query = tf.tile(query, multiples=[1, ads_key.shape[1]])
        query = tf.reshape(query, shape=[-1, ads_key.shape[1], ads_key.shape[2]])

        din_all_output = tf.concat([query, ads_key, query - ads_key, query * ads_key], axis=-1)

        att_hidden_units = att_hidden_units + [query_dim]

        att_layer_key = name

        if att_layer_key not in self.attention_layers_cache:
            att_layers = []
            for i in range(len(att_hidden_units)):
                att_dim = att_hidden_units[i]
                layer = tf.keras.layers.Dense(
                    units=att_dim,
                    activation=tf.nn.relu,
                    name=name+'_tower_'+str(i)
                )
                att_layers.append(layer)
            self.attention_layers_cache[att_layer_key] = att_layers

        att_layers = self.attention_layers_cache[att_layer_key]
        for i, layer in enumerate(att_layers):
            din_all_output = layer(din_all_output)

        key_dim = tf.shape(ads_key)[-1]
        key_dim_float = tf.cast(key_dim, tf.float32)
        scores = din_all_output / (key_dim_float ** 0.5)
        scores = tf.nn.sigmoid(scores)
        outputs = scores * tf.expand_dims(tf.cast(mask, din_all_output.dtype), 2)
        weighted_sum = outputs * ads_key
        weighted_sum = tf.reduce_sum(weighted_sum, axis=1)

        return weighted_sum

    def gate_dense(self, input, name, is_train, hidden_units, activations):
        bottom_layer_key = name

        if bottom_layer_key not in self.bottom_layers_cache:
            layers_dict = {}
            for i in range(len(hidden_units)):
                layer_dim = hidden_units[i]
                lay_act = activations[i]
                layer = tf.keras.layers.Dense(
                    units=layer_dim,
                    activation=lay_act,
                    name=name+'_bottom_layer_'+str(i),
                    kernel_regularizer=regularizers.l2(model_conf.l2_reg)
                )
                layers_dict['dense_layer_'+str(i)] = layer

            self.bottom_layers_cache[bottom_layer_key] = layers_dict

        cached_layers = self.bottom_layers_cache[bottom_layer_key]
        output = input
        for i in range(len(hidden_units)):
            layer = cached_layers['dense_layer_'+str(i)]
            output = layer(output)

        return output

    def task_tower(self, input, name, is_train, hidden_units, activations, dropout_dim):
        tower_layer_key = name

        if tower_layer_key not in self.task_layers_cache:
            layers_dict = {}
            for i in range(len(hidden_units)):
                layer_dim = hidden_units[i]
                lay_act = activations[i]
                layer = tf.keras.layers.Dense(
                    units=layer_dim,
                    activation=lay_act,
                    name=name+'_tower_layer_'+str(i),
                    kernel_regularizer=regularizers.l2(model_conf.l2_reg)
                )
                dropout_layer = None
                if i + 1 < len(dropout_dim) and dropout_dim[i + 1] > 1e-6:
                    dropout_layer = tf.keras.layers.Dropout(
                        rate=dropout_dim[i + 1],
                        name=name+'_'+str(i)+'_dropout'
                    )
                layers_dict['dense_layer_'+str(i)] = layer
                layers_dict['dropout_layer_'+str(i)] = dropout_layer

            if model_conf.use_bn:
                bn_layer = tf.keras.layers.BatchNormalization(name=name+'_bn')
                layers_dict['bn_layer'] = bn_layer

            self.task_layers_cache[tower_layer_key] = layers_dict

        cached_layers = self.task_layers_cache[tower_layer_key]
        output = input
        if 'bn_layer' in cached_layers:
            bn_layer = cached_layers['bn_layer']
            output = bn_layer(output, training=is_train)

        for i in range(len(hidden_units)):
            layer = cached_layers['dense_layer_'+str(i)]
            dropout_layer = cached_layers['dropout_layer_'+str(i)]
            output = layer(output)
            if dropout_layer is not None:
                output = dropout_layer(output, training=is_train)

        return output

    def call(self, inputs, training=None):
        sids, fids = inputs
        step = self.optimizer.iterations

        sid_list, fid_list = self.transform(sids, fids)

        pooled_output, slot_mask = self.process_and_pool_fused(sid_list, fid_list)

        #lr part
        lr_indices = self.slot_id_table.lookup(tf.constant(model_conf.lr_slot_ids, dtype=tf.dtypes.int32))
        lr_emb = tf.gather(pooled_output[:, :, 0], lr_indices, axis=1)
        lr = tf.reduce_mean(lr_emb, axis=1, keepdims=True)

        #fm part
        full_emb = pooled_output[:, :, 1:]
        square_sum_fm_embedding = tf.math.square(tf.reduce_mean(full_emb, 1))
        sum_square_fm_embedding = tf.reduce_mean(tf.math.square(full_emb), 1)
        fm = 0.5 * tf.math.subtract(square_sum_fm_embedding, sum_square_fm_embedding)

        #embedding part
        emb_slot_indices = self.slot_id_table.lookup(tf.constant(model_conf.embedding_slot_ids, dtype=tf.dtypes.int32))
        all_emb = tf.gather(pooled_output, emb_slot_indices, axis=1)
        all_emb = tf.reshape(all_emb, [tf.shape(all_emb)[0], -1])

        #din part
        query_slot_indices = self.slot_id_table.lookup(tf.constant(model_conf.query_slots, dtype=tf.dtypes.int32))
        query_input = tf.gather(pooled_output[:, :, 1:], query_slot_indices, axis=1)
        query_input = tf.reshape(query_input, [tf.shape(query_input)[0], -1])
        query_input = self.query_dense(query_input)

        pooled_output_v2, slot_mask_v2 = self.process_and_pool_fused(sid_list, fid_list, table_type='din_ads_table')
        ads_slot_indices = self.slot_id_table_din_ads.lookup(tf.constant(model_conf.ads_fea_slots, dtype=tf.dtypes.int32))
        ads_emb = tf.gather(pooled_output_v2, ads_slot_indices, axis=1)
        ads_emb = tf.reshape(ads_emb, [tf.shape(ads_emb)[0], -1])

        att_outputs = []
        for seq_name, seq_sid_ids in model_conf.seq_slot_dict.items():
            if seq_name == 'user_12h_click_seq':
                seq_slot_indices = self.slot_id_table_din_ads.lookup(tf.constant(seq_sid_ids, dtype=tf.dtypes.int32))
                seq_input = tf.gather(pooled_output_v2, seq_slot_indices, axis=1)
                seq_mask = tf.gather(slot_mask_v2, seq_slot_indices, axis=1)
            else:
                seq_slot_indices = self.slot_id_table.lookup(tf.constant(seq_sid_ids, dtype=tf.dtypes.int32))
                seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
                seq_mask = tf.gather(slot_mask, seq_slot_indices, axis=1)

            att_output = self.attention_din_ads(query_input, seq_input, seq_mask, ads_emb, seq_name, [50, 20])
            att_outputs.append(att_output)

        deep = tf.concat([all_emb] + att_outputs, axis=-1)

        lhuc_input = deep
        deep1 = self.lhuc_net(deep, [1024, 512, 128], 'deep_lhuc1', lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)
        deep2 = self.lhuc_net(deep, [1024, 512, 128], 'deep_lhuc2', lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)
        deep3 = self.lhuc_net(deep, [1024, 512, 128], 'deep_lhuc3', lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)
        deep4 = self.lhuc_net(deep, [1024, 512, 128], 'deep_lhuc4', lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)

        if self.summary_writer is not None and step % self.histogram_freq == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('layer/lr', tf.reduce_mean(lr), step=step)
                tf.summary.scalar('layer/fm', tf.reduce_mean(fm), step=step)
                tf.summary.scalar('layer/deep', tf.reduce_mean(deep), step=step)
                tf.summary.scalar('layer/deep1', tf.reduce_mean(deep1), step=step)
                tf.summary.scalar('layer/deep2', tf.reduce_mean(deep2), step=step)
                tf.summary.scalar('layer/deep3', tf.reduce_mean(deep3), step=step)
                tf.summary.scalar('layer/deep4', tf.reduce_mean(deep4), step=step)

        expert1 = tf.expand_dims(tf.concat([lr, fm, deep1], axis=1), axis=2)
        expert2 = tf.expand_dims(tf.concat([lr, fm, deep2], axis=1), axis=2)
        expert3 = tf.expand_dims(tf.concat([lr, fm, deep3], axis=1), axis=2)
        expert4 = tf.expand_dims(tf.concat([lr, fm, deep4], axis=1), axis=2)

        gate1 = tf.expand_dims(self.gate_dense(deep, "gate1", self.training, [128,4], [tf.nn.swish, tf.nn.softmax]), axis=1)
        gate2 = tf.expand_dims(self.gate_dense(deep, "gate2", self.training, [128,4], [tf.nn.swish, tf.nn.softmax]), axis=1)
        gate3 = tf.expand_dims(self.gate_dense(deep, "gate3", self.training, [128,4], [tf.nn.swish, tf.nn.softmax]), axis=1)
        gate4 = tf.expand_dims(self.gate_dense(deep, "gate4", self.training, [128,4], [tf.nn.swish, tf.nn.softmax]), axis=1)

        concat = tf.concat([expert1, expert2, expert3,expert4], axis=2)  # batch_size * expert_dim * expert_num

        buy_output = tf.reduce_sum(concat * gate1, axis=2)
        cat_output = tf.reduce_sum(concat * gate2, axis=2)
        clk_output = tf.reduce_sum(concat * gate3, axis=2)
        ext_output = tf.reduce_sum(concat * gate4, axis=2)

        buy_tower_output = self.task_tower(buy_output, "buy_tower", self.training, [128,32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)
        cat_tower_output = self.task_tower(cat_output, "cat_tower", self.training, [128,32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)
        click_tower_output = self.task_tower(clk_output, "click_tower", self.training, [128,32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)
        ext_tower_output = self.task_tower(ext_output, "ext_tower", self.training, [128,32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)

        buy_pred_org = self.dense_concat(buy_tower_output)
        cat_pred_org = self.dense_concat1(cat_tower_output)
        click_pred = self.dense_concat2(click_tower_output)
        ext_pred = self.dense_concat3(ext_tower_output)

        cat_pred = cat_pred_org
        buy_pred = buy_pred_org

        if self.is_save_model or self.pred:
            final_pred = buy_pred
            return final_pred, cat_pred, click_pred, ext_pred
        return buy_pred, cat_pred, click_pred, ext_pred,gate1, gate2, gate3,gate4