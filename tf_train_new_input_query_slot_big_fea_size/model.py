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

        # 保留这些 cache，lhuc/gate/task_tower 仍然用懒建层模式
        # 但 ads / attention 层在下面全部预建
        self.lhuc_layers_cache = {}
        self.bottom_layers_cache = {}
        self.task_layers_cache = {}

        self.loss = tf.keras.losses.binary_crossentropy
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            model_conf.learning_rate, decay_steps=model_conf.decay_steps, decay_rate=model_conf.decay_rate, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999,
            epsilon=1e-07, amsgrad=False, name='Adam')

        # ------------------------------------------------------------------ #
        # Embedding tables
        # ------------------------------------------------------------------ #
        self.emb_fm = tf.keras.layers.Embedding(
            model_conf.feature_size,
            model_conf.lr_emb_size + model_conf.fm_emb_size,
            embeddings_regularizer=regularizers.l2(model_conf.l2_reg))

        self.emb_din_ads = tf.keras.layers.Embedding(
            model_conf.ads_fea_size,
            model_conf.din_emb_size,
            embeddings_regularizer=regularizers.l2(model_conf.l2_reg))

        # ------------------------------------------------------------------ #
        # Hash / lookup tables
        # ------------------------------------------------------------------ #
        self.slot_id_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(model_conf.all_slot_ids, dtype=tf.int32),
                values=tf.range(len(model_conf.all_slot_ids), dtype=tf.int32)),
            default_value=-1)

        if fid_kv is not None:
            self.fid_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=fid_kv[0], values=fid_kv[1]),
                default_value=-1)
        else:
            self.fid_table = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int64, value_dtype=tf.int64,
                default_value=-1, empty_key=0, deleted_key=-1,
                initial_num_buckets=model_conf.num_buckets)
        self.counter = tf.Variable(1, dtype=tf.int64, trainable=False)

        self.slot_id_table_din_ads = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(model_conf.slot_id_v2, dtype=tf.int32),
                values=tf.range(len(model_conf.slot_id_v2), dtype=tf.int32)),
            default_value=-1)

        if fid_ads_kv is not None:
            self.fid_table_din_ads = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=fid_ads_kv[0], values=fid_ads_kv[1]),
                default_value=-1)
        else:
            self.fid_table_din_ads = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int64, value_dtype=tf.int64,
                default_value=-1, empty_key=0, deleted_key=-1,
                initial_num_buckets=model_conf.ads_num_buckets)
        self.counter_din_ads = tf.Variable(1, dtype=tf.int64, trainable=False)

        # ------------------------------------------------------------------ #
        # Shared layers
        # ------------------------------------------------------------------ #
        self.query_dense = tf.keras.layers.Dense(
            model_conf.fm_emb_size, activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(model_conf.l2_reg))

        self.dense_concat  = tf.keras.layers.Dense(1, activation="sigmoid",
            kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat1 = tf.keras.layers.Dense(1, activation="sigmoid",
            kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat2 = tf.keras.layers.Dense(1, activation="sigmoid",
            kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat3 = tf.keras.layers.Dense(1, activation="sigmoid",
            kernel_regularizer=regularizers.l2(model_conf.l2_reg))

        # ------------------------------------------------------------------ #
        # Pre-build ads_seq_cross_layer + attention_din_ads layers
        # for every sequence in seq_slot_dict (called in call()).
        #
        # Dimension analysis (all static Python ints):
        #   user_12h_click_seq  → table_type='din_ads_table' → emb dim = din_emb_size = 8
        #   other seqs          → table_type='emb_table'    → emb dim = fm_emb_size  = 8
        #
        # ads_output_dim == key_dim (== emb dim above) per the original call:
        #   ads_output_dim=tf.shape(key)[-1]  →  ads_output_dim=key_emb_dim
        #
        # att_hidden_units is hard-coded [50, 20] everywhere.
        # The final att layer appends query_dim (== fm_emb_size == 8), giving [50, 20, 8].
        # ------------------------------------------------------------------ #
        self._ads_layers   = {}   # seq_name -> {'weight_layers': [...], 'bias_layers': [...]}
        self._att_layers   = {}   # seq_name -> [Dense, ...]

        _att_hidden_units = [50, 20]
        _ads_hidden_dim   = 64

        for seq_name, seq_sid_ids in model_conf.seq_slot_dict.items():
            if seq_name == 'user_12h_click_seq':
                key_dim = model_conf.din_emb_size   # 8
            else:
                key_dim = model_conf.fm_emb_size    # 8

            ads_output_dim  = key_dim               # matches original: ads_output_dim=tf.shape(key)[-1]
            ads_input_dim   = key_dim               # key.shape[-1]
            layer_prefix    = seq_name + '_ads_layer'

            # ---- ads_seq_cross_layer weights ----
            ads_weight_dims = [_ads_hidden_dim, ads_input_dim * ads_output_dim]
            ads_weight_acts = [tf.nn.relu, None]
            weight_layers = []
            for i, (pdim, pact) in enumerate(zip(ads_weight_dims, ads_weight_acts)):
                weight_layers.append(tf.keras.layers.Dense(
                    units=pdim, activation=pact,
                    name=layer_prefix + '_ads_weight_' + str(i)))

            ads_bias_dims = [_ads_hidden_dim, ads_output_dim]
            bias_layers = []
            for i, pdim in enumerate(ads_bias_dims):
                bias_layers.append(tf.keras.layers.Dense(
                    units=pdim, activation=tf.nn.relu,
                    name=layer_prefix + '_ads_bias_' + str(i)))

            self._ads_layers[seq_name] = {
                'weight_layers': weight_layers,
                'bias_layers':   bias_layers,
                'ads_input_dim': ads_input_dim,
                'ads_output_dim': ads_output_dim,
            }

            # ---- attention tower layers ----
            # Original: att_hidden_units passed in as [50, 20],
            # then appended with query_dim inside attention_din_ads.
            # query_dim = tf.shape(query)[-1] = fm_emb_size (after query_dense) = 8
            query_dim = model_conf.fm_emb_size
            full_att_units = _att_hidden_units + [query_dim]  # [50, 20, 8]
            att_layers = []
            for i, att_dim in enumerate(full_att_units):
                att_layers.append(tf.keras.layers.Dense(
                    units=att_dim, activation=tf.nn.relu,
                    name=seq_name + '_tower_' + str(i)))
            self._att_layers[seq_name] = att_layers

        # Register all pre-built layers as model attributes so Keras tracks weights
        for seq_name in model_conf.seq_slot_dict:
            for i, layer in enumerate(self._ads_layers[seq_name]['weight_layers']):
                setattr(self, f'_ads_w_{seq_name}_{i}', layer)
            for i, layer in enumerate(self._ads_layers[seq_name]['bias_layers']):
                setattr(self, f'_ads_b_{seq_name}_{i}', layer)
            for i, layer in enumerate(self._att_layers[seq_name]):
                setattr(self, f'_att_{seq_name}_{i}', layer)

        # Cross Network — lazy init (built on first call)
        # self._cross_w = []
        # self._cross_b = []
        # self._cross_proj = None
        # self._cross_built = False
        # self._num_cross_layers = model_conf.num_cross_layers

    # ---------------------------------------------------------------------- #
    # Helper utilities
    # ---------------------------------------------------------------------- #

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

    def lhuc_net(self, nn_input, nn_dims, name, lhuc_input, lhuc_basic_dims,
                 dropout_dim, is_train, use_bn=False):
        cur_layer = nn_input
        nn_input_dim = nn_input.shape[-1]
        lhuc_dims = [nn_input_dim] + nn_dims[:-1]

        layer_key = name
        if layer_key not in self.lhuc_layers_cache:
            layers_dict = {}
            if use_bn:
                layers_dict['bn_layer'] = tf.keras.layers.BatchNormalization(
                    name=name + '_bn')
            for i in range(len(nn_dims)):
                lhuc_ds = lhuc_basic_dims + [lhuc_dims[i]]
                lhuc_layers = []
                for j in range(len(lhuc_ds)):
                    lhuc_layers.append(tf.keras.layers.Dense(
                        units=lhuc_ds[j], activation=tf.nn.relu,
                        name=name + '_' + str(i) + '_lhuc_' + str(j),
                        kernel_regularizer=regularizers.l2(0.001)))
                layers_dict['lhuc_layers_' + str(i)] = lhuc_layers
                layers_dict['dense_layer_' + str(i)] = tf.keras.layers.Dense(
                    units=nn_dims[i], activation=tf.nn.relu,
                    name=name + '_' + str(i) + '_dense',
                    kernel_regularizer=regularizers.l2(0.001))
                dropout_layer = None
                if i + 1 < len(dropout_dim) and dropout_dim[i + 1] > 1e-6:
                    dropout_layer = tf.keras.layers.Dropout(
                        rate=dropout_dim[i + 1],
                        name=name + '_' + str(i) + '_dropout')
                layers_dict['dropout_layer_' + str(i)] = dropout_layer
                layers_dict['ln_layer_' + str(i)] = tf.keras.layers.LayerNormalization(
                    name=name + '_' + str(i) + '_ln')
            self.lhuc_layers_cache[layer_key] = layers_dict

        cached = self.lhuc_layers_cache[layer_key]
        if use_bn and 'bn_layer' in cached:
            cur_layer = cached['bn_layer'](cur_layer, training=is_train)
        for i in range(len(nn_dims)):
            lhuc_output = lhuc_input
            for lhuc_layer in cached['lhuc_layers_' + str(i)]:
                lhuc_output = lhuc_layer(lhuc_output)
            lhuc_output = 1.0 + 0.5 * tf.nn.tanh(0.2 * lhuc_output)
            cur_layer = cur_layer * lhuc_output
            cur_layer = cached['dense_layer_' + str(i)](cur_layer)
            cur_layer = cached['ln_layer_' + str(i)](cur_layer)
            if cached['dropout_layer_' + str(i)] is not None:
                cur_layer = cached['dropout_layer_' + str(i)](
                    cur_layer, training=is_train)
        return cur_layer

    def fid_lookup_or_insert(self, valid_fids, table_type):
        flat_ids   = tf.reshape(valid_fids, [-1])
        unique_ids, idx_map = tf.unique(flat_ids)
        valid_mask = tf.not_equal(unique_ids, 0)
        unique_ids = tf.boolean_mask(unique_ids, valid_mask)

        if table_type == 'din_ads_table':
            unique_mapped = self.fid_table_din_ads.lookup(unique_ids)
        else:
            unique_mapped = self.fid_table.lookup(unique_ids)

        mask = tf.equal(unique_mapped, -1)

        def insert_and_update():
            miss_indices = tf.where(mask)
            miss_ids     = tf.gather_nd(unique_ids, miss_indices)
            num_new      = tf.shape(miss_ids)[0]
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
            return tf.tensor_scatter_nd_update(unique_mapped, miss_indices, new_values)

        final_mapped = tf.cond(
            tf.reduce_any(mask),
            true_fn=insert_and_update,
            false_fn=lambda: unique_mapped)
        return tf.reshape(tf.gather(final_mapped, idx_map), tf.shape(valid_fids))

    def fid_lookup(self, valid_fids, table_type):
        flat_ids   = tf.reshape(valid_fids, [-1])
        mapped_ids = (self.fid_table_din_ads.lookup(flat_ids)
                      if table_type == 'din_ads_table'
                      else self.fid_table.lookup(flat_ids))
        return tf.reshape(mapped_ids, tf.shape(valid_fids))

    def process_and_pool_fused(self, sid_list, fid_list, table_type='emb_table'):
        batch_size = tf.shape(sid_list)[0]
        n          = tf.shape(sid_list)[1]
        sid_flat   = tf.reshape(sid_list, [-1])
        fid_flat   = tf.reshape(fid_list, [-1])

        if table_type == 'din_ads_table':
            num_segments   = len(model_conf.slot_id_v2)
            mapped_indices = self.slot_id_table_din_ads.lookup(sid_flat)
        else:
            num_segments   = len(model_conf.all_slot_ids)
            mapped_indices = self.slot_id_table.lookup(sid_flat)

        mask          = tf.not_equal(mapped_indices, -1)
        valid_fids    = tf.boolean_mask(fid_flat, mask)
        valid_indices = tf.boolean_mask(mapped_indices, mask)

        batch_ids       = tf.repeat(tf.range(batch_size), n)
        valid_batch_ids = tf.boolean_mask(batch_ids, mask)
        combined        = valid_batch_ids * num_segments + valid_indices

        if self.is_save_model or self.pred:
            new_fid_list = self.fid_lookup(valid_fids, table_type)
        else:
            new_fid_list = self.fid_lookup_or_insert(valid_fids, table_type)

        vocab_size     = (model_conf.ads_fea_size if table_type == 'din_ads_table'
                          else model_conf.feature_size)
        valid_new_fids = (new_fid_list > 0) & (new_fid_list < vocab_size)
        new_fid_list   = tf.boolean_mask(new_fid_list, valid_new_fids)
        combined       = tf.boolean_mask(combined, valid_new_fids)

        embeds = (self.emb_din_ads(new_fid_list) if table_type == 'din_ads_table'
                  else self.emb_fm(new_fid_list))

        embedding_dim = tf.shape(embeds)[1]
        pooled_flat   = tf.math.unsorted_segment_mean(
            data=embeds, segment_ids=combined,
            num_segments=batch_size * num_segments)
        pooled_output = tf.reshape(pooled_flat, [batch_size, num_segments, embedding_dim])

        ones        = tf.ones_like(combined, dtype=tf.float32)
        counts_flat = tf.math.unsorted_segment_mean(
            data=ones, segment_ids=combined,
            num_segments=batch_size * num_segments)
        counts      = tf.reshape(counts_flat, [batch_size, num_segments])
        slot_mask   = tf.cast(tf.greater(counts, 0), dtype=tf.float32)

        return pooled_output, slot_mask

    # ---------------------------------------------------------------------- #
    # ads_seq_cross_layer — forward pass only, layers pre-built in __init__
    # ---------------------------------------------------------------------- #
    def _ads_seq_cross_forward(self, seq_name, nn_inputs):
        """
        Forward pass through ads_seq_cross_layer for the given sequence.
        All Dense layers were created in __init__ and stored in self._ads_layers.
        """
        info            = self._ads_layers[seq_name]
        ads_input_dim   = info['ads_input_dim']
        ads_output_dim  = info['ads_output_dim']
        weight_layers   = info['weight_layers']
        bias_layers     = info['bias_layers']

        ads_emb = nn_inputs  # Note: actual ads_emb is passed from attention_din_ads
        # (This method is not called directly; see _attention_din_ads_forward)
        raise RuntimeError("Call _attention_din_ads_forward instead.")

    # ---------------------------------------------------------------------- #
    # attention_din_ads — forward pass only, layers pre-built in __init__
    # ---------------------------------------------------------------------- #
    def _attention_din_ads_forward(self, query, key, mask, ads_emb, seq_name):
        """
        Forward pass combining ads_seq_cross_layer + attention tower.
        No layer creation here — everything is looked up from self._ads_layers
        and self._att_layers.
        """
        info          = self._ads_layers[seq_name]
        ads_input_dim = info['ads_input_dim']
        ads_output_dim= info['ads_output_dim']
        weight_layers = info['weight_layers']
        bias_layers   = info['bias_layers']

        # ---- ads_seq_cross_layer forward ----
        ads_weight = ads_emb
        for layer in weight_layers:
            ads_weight = layer(ads_weight)
        ads_weight = tf.reshape(ads_weight, [-1, ads_input_dim, ads_output_dim])

        ads_bias = ads_emb
        for layer in bias_layers:
            ads_bias = layer(ads_bias)
        ads_bias = tf.expand_dims(ads_bias, 1)

        ads_key = tf.matmul(key, ads_weight)
        ads_key = tf.add(ads_key, ads_bias)

        # ---- attention tower forward ----
        query_dim = tf.shape(query)[-1]
        query_tiled = tf.tile(query, multiples=[1, ads_key.shape[1]])
        query_tiled = tf.reshape(query_tiled,
                                 shape=[-1, ads_key.shape[1], ads_key.shape[2]])

        din_all = tf.concat(
            [query_tiled, ads_key, query_tiled - ads_key, query_tiled * ads_key],
            axis=-1)

        att_layers = self._att_layers[seq_name]
        for layer in att_layers:
            din_all = layer(din_all)

        key_dim       = tf.shape(ads_key)[-1]
        key_dim_float = tf.cast(key_dim, tf.float32)
        scores        = din_all / (key_dim_float ** 0.5)
        scores        = tf.nn.sigmoid(scores)
        outputs       = scores * tf.expand_dims(
            tf.cast(mask, din_all.dtype), 2)
        weighted_sum  = tf.reduce_sum(outputs * ads_key, axis=1)

        return weighted_sum

    def gate_dense(self, input, name, is_train, hidden_units, activations):
        layer_key = name
        if layer_key not in self.bottom_layers_cache:
            layers_dict = {}
            for i in range(len(hidden_units)):
                layers_dict['dense_layer_' + str(i)] = tf.keras.layers.Dense(
                    units=hidden_units[i], activation=activations[i],
                    name=name + '_bottom_layer_' + str(i),
                    kernel_regularizer=regularizers.l2(model_conf.l2_reg))
            self.bottom_layers_cache[layer_key] = layers_dict
        cached = self.bottom_layers_cache[layer_key]
        output = input
        for i in range(len(hidden_units)):
            output = cached['dense_layer_' + str(i)](output)
        return output

    def task_tower(self, input, name, is_train, hidden_units, activations, dropout_dim):
        layer_key = name
        if layer_key not in self.task_layers_cache:
            layers_dict = {}
            for i in range(len(hidden_units)):
                layers_dict['dense_layer_' + str(i)] = tf.keras.layers.Dense(
                    units=hidden_units[i], activation=activations[i],
                    name=name + '_tower_layer_' + str(i),
                    kernel_regularizer=regularizers.l2(model_conf.l2_reg))
                dropout_layer = None
                if i + 1 < len(dropout_dim) and dropout_dim[i + 1] > 1e-6:
                    dropout_layer = tf.keras.layers.Dropout(
                        rate=dropout_dim[i + 1],
                        name=name + '_' + str(i) + '_dropout')
                layers_dict['dropout_layer_' + str(i)] = dropout_layer
            if model_conf.use_bn:
                layers_dict['bn_layer'] = tf.keras.layers.BatchNormalization(
                    name=name + '_bn')
            self.task_layers_cache[layer_key] = layers_dict
        cached = self.task_layers_cache[layer_key]
        output = input
        if 'bn_layer' in cached:
            output = cached['bn_layer'](output, training=is_train)
        for i in range(len(hidden_units)):
            output = cached['dense_layer_' + str(i)](output)
            if cached['dropout_layer_' + str(i)] is not None:
                output = cached['dropout_layer_' + str(i)](output, training=is_train)
        return output

    # def build_cross(self, deep_dim):
    #     for i in range(self._num_cross_layers):
    #         w = tf.Variable(
    #             tf.keras.initializers.GlorotUniform()(shape=[deep_dim, 1]),
    #             trainable=True, name=f'cross_w_{i}', dtype=tf.float32)
    #         b = tf.Variable(
    #             tf.zeros([deep_dim]),
    #             trainable=True, name=f'cross_b_{i}', dtype=tf.float32)
    #         setattr(self, f'_cross_w_{i}', w)
    #         setattr(self, f'_cross_b_{i}', b)
    #         self._cross_w.append(w)
    #         self._cross_b.append(b)
    #     self._cross_proj = tf.keras.layers.Dense(
    #         deep_dim, activation=tf.nn.relu, name='cross_proj',
    #         kernel_regularizer=regularizers.l2(model_conf.l2_reg))
    #     self._cross_built = True

    # ---------------------------------------------------------------------- #
    # call — forward pass only, zero layer creation
    # ---------------------------------------------------------------------- #
    def call(self, inputs, training=None):
        sids, fids = inputs
        step = self.optimizer.iterations

        sid_list, fid_list = self.transform(sids, fids)

        pooled_output, slot_mask = self.process_and_pool_fused(sid_list, fid_list)

        # lr part
        lr_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.lr_slot_ids, dtype=tf.dtypes.int32))
        lr_emb = tf.gather(pooled_output[:, :, 0], lr_indices, axis=1)
        lr     = tf.reduce_mean(lr_emb, axis=1, keepdims=True)

        # fm part
        full_emb              = pooled_output[:, :, 1:]
        square_sum_fm         = tf.math.square(tf.reduce_mean(full_emb, 1))
        sum_square_fm         = tf.reduce_mean(tf.math.square(full_emb), 1)
        fm                    = 0.5 * tf.math.subtract(square_sum_fm, sum_square_fm)

        # embedding part
        emb_slot_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.embedding_slot_ids, dtype=tf.dtypes.int32))
        all_emb          = tf.gather(pooled_output, emb_slot_indices, axis=1)
        num_emb_slots    = len(model_conf.embedding_slot_ids)
        lr_fm_dim        = model_conf.lr_emb_size + model_conf.fm_emb_size
        all_emb          = tf.reshape(all_emb, [-1, num_emb_slots * lr_fm_dim])

        # query part
        query_slot_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.query_slots, dtype=tf.dtypes.int32))
        query_input        = tf.gather(pooled_output[:, :, 1:], query_slot_indices, axis=1)
        num_query_slots    = len(model_conf.query_slots)
        fm_emb_dim         = model_conf.fm_emb_size
        query_input        = tf.reshape(query_input, [-1, num_query_slots * fm_emb_dim])
        query_input        = self.query_dense(query_input)

        # ads embedding part
        pooled_output_v2, slot_mask_v2 = self.process_and_pool_fused(
            sid_list, fid_list, table_type='din_ads_table')
        ads_slot_indices = self.slot_id_table_din_ads.lookup(
            tf.constant(model_conf.ads_fea_slots, dtype=tf.dtypes.int32))
        ads_emb          = tf.gather(pooled_output_v2, ads_slot_indices, axis=1)
        num_ads_slots    = len(model_conf.ads_fea_slots)
        din_emb_dim      = model_conf.din_emb_size
        ads_emb          = tf.reshape(ads_emb, [-1, num_ads_slots * din_emb_dim])

        # DIN attention — pure forward pass, no layer creation
        att_outputs = []
        for seq_name, seq_sid_ids in model_conf.seq_slot_dict.items():
            if seq_name == 'user_12h_click_seq':
                seq_slot_indices = self.slot_id_table_din_ads.lookup(
                    tf.constant(seq_sid_ids, dtype=tf.dtypes.int32))
                seq_input = tf.gather(pooled_output_v2, seq_slot_indices, axis=1)
                seq_mask  = tf.gather(slot_mask_v2, seq_slot_indices, axis=1)
            else:
                seq_slot_indices = self.slot_id_table.lookup(
                    tf.constant(seq_sid_ids, dtype=tf.dtypes.int32))
                seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
                seq_mask  = tf.gather(slot_mask, seq_slot_indices, axis=1)

            att_output = self._attention_din_ads_forward(
                query_input, seq_input, seq_mask, ads_emb, seq_name)
            att_outputs.append(att_output)

        deep = tf.concat([all_emb] + att_outputs, axis=-1)

        # # Cross Network
        # if not self._cross_built:
        #     self.build_cross(int(deep.shape[-1]))
        # x0 = deep
        # xl = deep
        # for i in range(self._num_cross_layers):
        #     xl_w = tf.matmul(xl, self._cross_w[i])
        #     xl = x0 * xl_w + self._cross_b[i] + xl
        # deep_combined = tf.concat([xl, deep], axis=-1)
        # deep_combined = self._cross_proj(deep_combined)

        deep_combined = deep
        # LHUC towers
        lhuc_input = deep_combined
        deep1 = self.lhuc_net(deep_combined, [1024, 512, 128], 'deep_lhuc1',
                               lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)
        deep2 = self.lhuc_net(deep_combined, [1024, 512, 128], 'deep_lhuc2',
                               lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)
        deep3 = self.lhuc_net(deep_combined, [1024, 512, 128], 'deep_lhuc3',
                               lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)
        deep4 = self.lhuc_net(deep_combined, [1024, 512, 128], 'deep_lhuc4',
                               lhuc_input, [128], self.dropout_dim, self.training, model_conf.use_bn)

        if self.summary_writer is not None and step % self.histogram_freq == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('layer/lr',    tf.reduce_mean(lr),    step=step)
                tf.summary.scalar('layer/fm',    tf.reduce_mean(fm),    step=step)
                tf.summary.scalar('layer/deep',  tf.reduce_mean(deep),  step=step)
                tf.summary.scalar('layer/deep1', tf.reduce_mean(deep1), step=step)
                tf.summary.scalar('layer/deep2', tf.reduce_mean(deep2), step=step)
                tf.summary.scalar('layer/deep3', tf.reduce_mean(deep3), step=step)
                tf.summary.scalar('layer/deep4', tf.reduce_mean(deep4), step=step)

        # MoE experts + gates
        expert1 = tf.expand_dims(tf.concat([lr, fm, deep1], axis=1), axis=2)
        expert2 = tf.expand_dims(tf.concat([lr, fm, deep2], axis=1), axis=2)
        expert3 = tf.expand_dims(tf.concat([lr, fm, deep3], axis=1), axis=2)
        expert4 = tf.expand_dims(tf.concat([lr, fm, deep4], axis=1), axis=2)

        gate1 = tf.expand_dims(self.gate_dense(deep_combined, "gate1", self.training,
                               [128, 4], [tf.nn.swish, tf.nn.softmax]), axis=1)
        gate2 = tf.expand_dims(self.gate_dense(deep_combined, "gate2", self.training,
                               [128, 4], [tf.nn.swish, tf.nn.softmax]), axis=1)
        gate3 = tf.expand_dims(self.gate_dense(deep_combined, "gate3", self.training,
                               [128, 4], [tf.nn.swish, tf.nn.softmax]), axis=1)
        gate4 = tf.expand_dims(self.gate_dense(deep_combined, "gate4", self.training,
                               [128, 4], [tf.nn.swish, tf.nn.softmax]), axis=1)

        concat = tf.concat([expert1, expert2, expert3, expert4], axis=2)

        buy_output   = tf.reduce_sum(concat * gate1, axis=2)
        cat_output   = tf.reduce_sum(concat * gate2, axis=2)
        clk_output   = tf.reduce_sum(concat * gate3, axis=2)
        ext_output   = tf.reduce_sum(concat * gate4, axis=2)

        buy_tower    = self.task_tower(buy_output,  "buy_tower",   self.training,
                                       [128, 32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)
        cat_tower    = self.task_tower(cat_output,  "cat_tower",   self.training,
                                       [128, 32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)
        click_tower  = self.task_tower(clk_output,  "click_tower", self.training,
                                       [128, 32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)
        ext_tower    = self.task_tower(ext_output,  "ext_tower",   self.training,
                                       [128, 32], [tf.nn.relu, tf.nn.relu], self.dropout_dim)

        buy_pred     = self.dense_concat(buy_tower)
        cat_pred     = self.dense_concat1(cat_tower)
        click_pred   = self.dense_concat2(click_tower)
        ext_pred     = self.dense_concat3(ext_tower)

        if self.is_save_model or self.pred:
            return buy_pred, cat_pred, click_pred, ext_pred
        return buy_pred, cat_pred, click_pred, ext_pred, gate1, gate2, gate3, gate4
