import tensorflow as tf
# This module contains pooling strategies to convert multiple
# item embeddings into fixed dimension

# All the strategies should be registered into this dictionary
_pooling_strategies = dict()


def strategy(name):
    """
    Decorator builder to register pooling strategy
    :param name: Name of the strategy
    :return: decorator which helps register pooling function
    """
    def _wrapper(func):
        if name in _pooling_strategies:
            raise KeyError("Duplicated pooling strategy: {}".format(name))
        _pooling_strategies[name] = func
        return func

    return _wrapper


def pool_embeddings(strategy_name, log_embeddings, log_masks, trainable, **kwargs):
    if strategy_name not in _pooling_strategies:
        raise KeyError("Unknown pooling strategy: {}".format(strategy_name))

    return _pooling_strategies[strategy_name](log_embeddings, log_masks, trainable, **kwargs)


# Utility functions
def _do_layer_norm(outputs, name="FuseTransformerOutput"):
    outputs = tf.layers.dense(
        outputs,
        units=int(outputs.shape[-1]),
        activation=None,
        name=name,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.constant_initializer(0)
    )
    return tf.contrib.layers.layer_norm(outputs, reuse=tf.AUTO_REUSE, scope="norm-pooled-embed")


def _do_direct_layer_norm(outputs, name="direct-layer-norm"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.contrib.layers.layer_norm(outputs, reuse=tf.AUTO_REUSE, scope="layer_norm")


# Pooling strategies
@strategy("max")
def _max_pooling(log_embeddings, log_masks, trainable=True):
    # Insert CLS fake symbol
    batch_size = tf.shape(log_embeddings)[0]
    dim_log = log_embeddings.shape[2]

    log_masks = tf.concat(
        [tf.ones([batch_size, 1], tf.float32), log_masks],
        axis=1
    )

    cls_embedding = tf.get_variable(
        name="CLS_embedding", dtype=tf.float32, shape=[1, 1, dim_log],
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable
    )
    log_embeddings = tf.concat([tf.tile(cls_embedding, [batch_size, 1, 1]), log_embeddings], axis=1)

    embeddings = log_embeddings + tf.reshape(-1e30 * (1 - log_masks), [-1, log_embeddings.shape[1], 1])
    return tf.reduce_max(embeddings, axis=1)


@strategy("mean")
def _mean_pooling(log_embeddings, log_masks, trainable=True):
    print("Mean pooling is enabled")
    log_embeddings = tf.multiply(tf.expand_dims(log_masks, axis=-1), log_embeddings)
    query_count = tf.maximum(1.0, tf.reduce_sum(log_masks, axis=-1, keep_dims=True))
    print("QueryCount:", query_count)
    return tf.reduce_sum(log_embeddings, axis=1) / query_count


@strategy("mhop")
def _multihop(log_embeddings, log_masks, trainable=True):
    num_hops = 5
    batch_size = tf.shape(log_embeddings)[0]
    d_model = log_embeddings.shape[-1]

    # Extract keys & values
    keys = _do_direct_layer_norm(tf.layers.dense(
        log_embeddings,
        d_model,
        use_bias=None,
        name="extract_keys",
        activation=None
    ), name="ln-keys")  # (N, T_k, d_model)
    values = _do_direct_layer_norm(tf.layers.dense(
        log_embeddings,
        d_model,
        use_bias=False,
        activation=None,
        name="extract_values"
    ), name="ln-values")
    values = tf.multiply(tf.expand_dims(log_masks, axis=-1), values)

    # Prepare the outputs
    state_sum = tf.get_variable(
        name="state_embedding", dtype=tf.float32, shape=[1, 1, d_model],
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable
    )
    state_sum = tf.tile(state_sum, [batch_size, 1, 1])
    states = state_sum

    multi_step_states = []
    for i in range(num_hops):
        scores = tf.matmul(keys, states, transpose_b=True)
        print("Scores:", scores)
        scores = tf.squeeze(scores, axis=-1) + (1 - log_masks) * -1e8
        print("Masked scores:", scores)
        atten_weights = tf.expand_dims(tf.nn.softmax(scores, axis=-1), axis=-1)
        print("Atten weights:", atten_weights)
        state_delta = tf.matmul(atten_weights, values, transpose_a=True)
        print("state delta", state_delta)

        state_sum += state_delta
        states = _do_direct_layer_norm(state_sum, "ln_states")
        multi_step_states.append(tf.squeeze(states, axis=1))

    return multi_step_states


@strategy("gruhop")
def _gruhop(log_embeddings, log_masks, trainable=True):
    num_hops = 5
    num_max_items = int(log_embeddings.shape[1])
    batch_size = tf.shape(log_embeddings)[0]
    d_model = log_embeddings.shape[-1]

    # Extract keys & values
    state_list = []

    items = log_embeddings
    cell = tf.nn.rnn_cell.GRUCell(d_model, reuse=tf.AUTO_REUSE,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.constant_initializer(0))

    gru_state = tf.get_variable(
        name="init_state", dtype=tf.float32, shape=[1, d_model],
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable,
    )
    gru_state = tf.tile(tf.nn.tanh(gru_state), [batch_size, 1])

    #state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    for i in range(num_hops):
        score_mlp_input = tf.concat(
            [items, tf.tile(tf.expand_dims(gru_state, -2), [1,num_max_items,1])],
            axis=-1
        )
        scores = tf.layers.dense(
            tf.layers.dense(
                score_mlp_input,
                d_model,
                use_bias=True,
                activation=tf.nn.tanh,
                name="score_l1",
                reuse=tf.AUTO_REUSE
            ),
            units=1,
            activation=None,
            name="score_l2",
            reuse=tf.AUTO_REUSE
        )
        print("scores:", scores)
        scores = tf.squeeze(scores, axis=-1) + (1 - log_masks) * -1e8
        atten_weights = tf.expand_dims(tf.nn.softmax(scores, axis=-1), axis=-1)
        print("Atten weights:", atten_weights)
        gru_input = tf.squeeze(tf.matmul(atten_weights, items, transpose_a=True), axis=1)
        print("state delta", gru_input)

        gru_output, gru_state = cell(gru_input, gru_state)
        state_list.append(gru_output)

    return state_list
