# -*- coding: utf-8 -*-#
import tensorflow as tf


class OdpsDataLoader:
    def __init__(self, table_name, max_query_per_week, max_words_per_query, target_length, mode, repeat=None, batch_size=128, shuffle=2000, slice_id=0, slice_count=1, weighted_target=0):
        # Avoid destroying input parameter
        self._table_name = table_name
        self._max_query_per_week = max_query_per_week
        self._max_words_per_query = max_words_per_query
        self._target_length = target_length
        self._slice_id = slice_id
        self._slice_count = slice_count
        self._batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._mode = mode
        self._weighted_target = weighted_target

    def _parse_words(self, text, max_length):
        word_strs = tf.string_split([text], " ")
        return tf.string_to_number(word_strs.values, out_type=tf.int64)[:max_length], tf.minimum(tf.shape(word_strs)[-1], max_length)

    def _parse_indices(self, text):
        indices = tf.string_split([text], " ")
        return tf.string_to_number(indices.values, out_type=tf.int64)

    def _train_data_parser(self, id, history, target_weights, query_idx, word_idx, target_words):
        history = self._parse_indices(history)
        query_idx = self._parse_indices(query_idx)
        word_idx = self._parse_indices(word_idx)
        word_coord = tf.stack([query_idx, word_idx], axis=1)
        print("Word coord:", word_coord)

        history_word_tensor = tf.SparseTensor(
            indices=word_coord,
            values=history + 1,
            dense_shape=[self._max_query_per_week, self._max_words_per_query]
        )
        history_words = tf.sparse_tensor_to_dense(history_word_tensor, validate_indices=False)

        target_words, target_len = self._parse_words(target_words, self._target_length)
        target_words = tf.pad(target_words, [[0, self._target_length - target_len]], "CONSTANT")

        result = dict()
        if self._weighted_target:
            target_weights, _ = self._parse_words(target_weights, self._target_length)
            target_weights = tf.log(1 + tf.cast(target_weights, tf.float32))
            target_weights = target_weights / tf.reduce_sum(target_weights)
            target_weights = tf.pad(target_weights, [[0, self._target_length - target_len]], "CONSTANT", constant_values=0.)
            result["target_weights"] = target_weights

        result.update({
            "id": id,
            "history_words": tf.reshape(history_words, [-1, self._max_words_per_query]),
            "target_words": target_words,
            "target_len": target_len
        })

        return result, tf.constant(0, dtype=tf.int32) # fake label

    def _test_data_parser(self, id, history, _, query_idx, word_idx):
        history = self._parse_indices(history)
        query_idx = self._parse_indices(query_idx)
        word_idx = self._parse_indices(word_idx)
        word_coord = tf.stack([query_idx, word_idx], axis=1)
        print("Word coord:", word_coord)

        history_word_tensor = tf.SparseTensor(
            indices=word_coord,
            values=history + 1,
            dense_shape=[self._max_query_per_week, self._max_words_per_query]
        )
        history_words = tf.sparse_tensor_to_dense(history_word_tensor, validate_indices=False)

        return {
            "id": id,
            "history_words": tf.reshape(history_words, [-1, self._max_words_per_query]),
            "history_length": tf.shape(history)[-1]
        }, tf.constant(0, dtype=tf.int32)  # fake label

    def _train_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=[""] * 6,
                slice_id=self._slice_id,
                slice_count=self._slice_count
            )

            dataset = dataset.map(self._train_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.dataset = dataset.batch(self._batch_size)
            return dataset

    def _test_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=[""] * 5,
                slice_id=self._slice_id,
                slice_count=self._slice_count
            )

            dataset = dataset.map(self._test_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.dataset = dataset.batch(self._batch_size)
            return dataset

    def input_fn(self):
        return self._train_data_fn() if self._mode is tf.estimator.ModeKeys.TRAIN else self._test_data_fn()