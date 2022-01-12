# -*- coding: UTF-8 -*-
from collections import namedtuple
import collections
import re
from argparse import Namespace
import tensorflow as tf
from tensorflow.contrib import layers
from util import diagnose
from model import pooling


try:
    SparseTensor = tf.sparse.SparseTensor
    to_dense = tf.sparse.to_dense
except:
    SparseTensor = tf.SparseTensor
    to_dense = tf.sparse_tensor_to_dense


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    #if name not in name_to_variable or ("mlp" in name or "/beta" in name or "/gamma" in name):
    if name not in name_to_variable or (not name.startswith("encoder") and not name.startswith("transformer/num_blocks")):
        print("SKIP variable {}".format(name))
        continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


class _AveragePooling:
    def __init__(self, **kwargs):
        pass

    def __call__(self, embeddings, masks):
        # batch * length * 1
        multiplier = tf.expand_dims(masks, axis=-1)
        embeddings_sum = tf.reduce_sum(
            tf.multiply(multiplier, embeddings),
            axis=-2
        )
        length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=-1), 1.0), axis=-1)
        embedding_avg = embeddings_sum / length
        return embedding_avg


class _WordEmbeddingEncoder:
    def __init__(self, word_count, dimension, training, scope_name, *args, **kwargs):
        training = training
        self._scope_name = scope_name
        self._word_dimension = dimension
        self._average_pooling = _AveragePooling()

        with tf.variable_scope(scope_name, reuse=False):
            self._word_embedding = tf.get_variable(
                "WordEmbedding",
                [word_count, dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=training
            )
            # zeros padding
            self._word_embedding = tf.concat((tf.zeros(shape=[1, dimension]),self._word_embedding[1:, :]), 0)

    def __call__(self, content_words, content_mask):
        """
        embeddings of behaviors
        :param content_words: batch_size * num_queries * num_words
        :param content_mask: batch_size * num_queries * num_words
        :return:
        """
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            print("Content words:", content_words, content_words.shape)
            word_embedding = tf.nn.embedding_lookup(self._word_embedding, content_words)

            print("word embedding before pooling:", word_embedding)
            word_embedding = self._average_pooling(word_embedding, content_mask)
            print("avg_embedding shape is ", word_embedding.shape)
            return word_embedding


class _MlpTransformer(object):
    def __init__(self, layers, dropout, training, scope_name):
        self._layers = layers
        self._training = training
        self._dropout = dropout
        self._scope = tf.variable_scope(scope_name)

    def __call__(self, input):
        with self._scope as scope:
            values = input
            for i, n_units in enumerate(self._layers[:-1], 1):
                if self._training and self._dropout > 0:
                    print("In training mode, use dropout")
                    values = tf.nn.dropout(values, keep_prob=1-self._dropout)

                with tf.variable_scope("mlp-layer-%d" % i) as hidden_layer_scope:
                    values = layers.fully_connected(
                        values, num_outputs=n_units, activation_fn=tf.nn.relu,
                        scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
                    )

            return layers.linear(
                values, self._layers[-1], scope=scope, reuse=tf.AUTO_REUSE
            )


class _TransformerEncoder:
    def __init__(self, training, scope_name, model_dim, pooling):
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)
        self.training = training
        self.model_dim = model_dim
        self._pooling = pooling

    def __call__(self, log_embeddings, log_masks):
        """
        :param log_embeddings: BatchSize * N_QUERIES * N_DIMENSION
        :param log_masks: BatchSize * N_QUERIES
        :return:
        """
        print("Log Embedding:", log_embeddings, " Log Masks:", log_masks)
        with self._scope:
            #num_queries_per_week = log_embeddings.shape[1]
            # Build transformer encoder
            with tf.variable_scope("num_blocks_{}".format(0), reuse=tf.AUTO_REUSE):
                # feed forward
                #log_embeddings = ff(log_embeddings, num_units=[self.feed_forward_in_dim, self.model_dim])
                log_embeddings = tf.layers.dense(
                    log_embeddings,
                    self.model_dim,
                    activation=tf.nn.relu,
                    name="ff-layer",
                    reuse=tf.AUTO_REUSE
                )
                # log_embeddings = tf.contrib.layers.layer_norm(
                #     log_embeddings,
                #     reuse=tf.AUTO_REUSE,
                #     scope="layer_norm"
                # )

            #log_embeddings = tf.multiply(tf.expand_dims(log_masks, axis=-1), log_embeddings)
            # shape(?,300,512)
            return pooling.pool_embeddings(self._pooling, log_embeddings, log_masks, trainable=self.training)


class TextTransformerNet:
    ModelConfigs = namedtuple("ModelConfigs", ("dropout_rate", "num_vocabulary", "mlp_layers", "model_dim",
                                               "train_batch_size", "max_query_per_week", "max_words_per_query",
                                               "word_emb_dim", "pooling"))

    def __init__(self, model_configs, train_configs, predict_configs, run_configs):
        self._model_configs = model_configs
        self._train_configs = train_configs
        self._predict_configs = predict_configs
        self._run_configs = run_configs

    def init_params(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        (assignment_map, initialized_variable_names) = \
            get_assignment_map_from_checkpoint(tvars, self._train_configs.init_checkpoint)
        tf.train.init_from_checkpoint(self._train_configs.init_checkpoint, assignment_map)

        tf.logging.info("**** initialized_variable_names  **** \n %s" % initialized_variable_names)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

    def _train(self, model_output, labels):
        if self._train_configs.init_checkpoint:
            self.init_params()

        max_gradient_norm = None
        if self._train_configs.max_grad_norm > 0:
            max_gradient_norm = self._train_configs.max_grad_norm
        tf.logging.info("Gradient clipping is turned {}".format("ON" if max_gradient_norm else "OFF"))

        train_op = tf.contrib.layers.optimize_loss(
            loss=model_output.loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self._train_configs.learning_rate,
            optimizer="Adam",
            summaries=[
                "loss"
            ],
            clip_gradients=max_gradient_norm
        )

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=model_output.loss,
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "loss": model_output.loss,
                        "loss_last_step": model_output.loss_last_step
                        # "accuracy": 100. * tf.reduce_mean(tf.cast(tf.equal(tf.cast(model_output.prediction,tf.int32), labels), tf.float32))
                    },
                    every_n_iter=self._run_configs.log_every
                ),
                diagnose.GraphPrinterHook()#,
                # tf.train.ProfilerHook(
                #     save_secs=60,
                #     show_memory=True,
                #     output_dir="oss://wf135777-lab/profile/"
                # )
            ]
        )

    def _predict(self, features, model_output):
        outputs = dict(
            oneid=model_output.oneid,
            feature=tf.reduce_join(
                tf.as_string(model_output.feature, shortest=True, scientific=False),
                axis=1,
                separator=self._predict_configs.separator
            ),
            length=features["history_length"]
        )
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=outputs
        )

    def _get_target_dist(self, target, target_len, target_weights, batch_size, vocab_size):
        non_zero_indices = tf.where(tf.not_equal(target, 0))
        col_indices = tf.cast(tf.gather_nd(target, non_zero_indices), tf.int64)

        if not self._train_configs.weighted_target:
            print("Using equal weight for all target words")
            target_weights = tf.ones([tf.shape(non_zero_indices)[0]], dtype=tf.float32)
        else:
            print("Using specified target weights")
            target_weights = tf.gather_nd(target_weights, non_zero_indices)

        expanded_target = to_dense(
            SparseTensor(
                indices=tf.concat([
                    tf.reshape(non_zero_indices[:, 0], [-1, 1]),
                    tf.reshape(col_indices, [-1, 1]),
                ], axis=1),
                values=target_weights,
                dense_shape=[batch_size, vocab_size]
            )
        )

        if not self._train_configs.weighted_target:
            expanded_target = expanded_target / tf.cast(tf.reshape(target_len, [-1, 1]), tf.float32)

        return expanded_target

    def _build_model(self, features, labels, mode):
        training = mode is tf.estimator.ModeKeys.TRAIN

        history_text = features.get("history_words")
        history_mask = tf.cast(tf.not_equal(history_text, 0), tf.float32)
        print("History text:", history_text)

        # 1. Convert each query into a single embedding
        word_encoder = _WordEmbeddingEncoder(
            scope_name="encoder",
            word_count=self._model_configs.num_vocabulary,
            dimension=self._model_configs.word_emb_dim,
            training=training
        )
        # Batch * N_Queries * D_Embedding
        history_text_embedding = word_encoder(
            history_text,
            history_mask
        )

        # 2. aggregation of embeddings of behaviors
        transformer_enc = _TransformerEncoder(
                               pooling=self._model_configs.pooling,
                               training=training,
                               scope_name="transformer",
                               model_dim=self._model_configs.model_dim)
        user_embedding = transformer_enc(history_text_embedding, history_mask[:, :, 0])

        if mode is tf.estimator.ModeKeys.PREDICT:
            # Do inference
            if type(user_embedding) is list:
                if self._predict_configs.hop > 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("You choose the {}/{} output".format(
                        self._predict_configs.hop, len(user_embedding)
                    ))
                    user_embedding = user_embedding[self._predict_configs.hop-1]
                else:
                    user_embedding = user_embedding[-1]

            return Namespace(
                oneid=features["oneid"],
                feature=user_embedding
            )
        else:
            # Build loss
            loss = 0
            loss_cur_step = 0
            target_dist = self._get_target_dist(
                target=features["target_words"],
                target_len=features["target_len"],
                target_weights=features.get("target_weights"),
                batch_size=self._model_configs.train_batch_size,
                vocab_size=self._model_configs.num_vocabulary
            )

            mlp_transformer = _MlpTransformer(
                layers=[int(x) for x in self._model_configs.mlp_layers.split(",")] + [self._model_configs.num_vocabulary],
                dropout=self._model_configs.dropout_rate,
                training=training,
                scope_name="mlp"
            )

            embedding_list = user_embedding
            if type(embedding_list) is not list:
                embedding_list = [embedding_list]

            for step in range(len(embedding_list)):
                print("Build loss for step #{}".format(step))
                supervised_embedding = embedding_list[step]
                logits = mlp_transformer(supervised_embedding)
                loss_cur_step = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits,
                        labels=tf.stop_gradient(target_dist)
                    )
                )
                loss += 1. / len(embedding_list) * loss_cur_step
                if len(embedding_list) > 1:
                    tf.summary.scalar("loss_step_{}".format(step), loss_cur_step)

            return Namespace(
                loss=loss,
                loss_last_step = loss_cur_step
            )

    def model_fn(self, features, labels, mode):
        model_output = self._build_model(features, labels, mode)

        if mode is tf.estimator.ModeKeys.TRAIN:
            return self._train(model_output, labels)
        elif mode is tf.estimator.ModeKeys.PREDICT:
            return self._predict(features, model_output)
        elif mode is tf.estimator.ModeKeys.EVAL:
            raise ValueError("Not implemented")
