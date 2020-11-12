import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  #dk = tf.shape(tf.shape(k)[-1])
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
    
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = tf.cast(q, tf.float32)
    k = tf.cast(k, tf.float32)
    v = tf.cast(v, tf.float32)

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    # concat_attention.shape: (1, 512, 464)

    input_shape = concat_attention.shape
    output = tf.keras.layers.Conv1D(32, 1, activation='relu', input_shape=input_shape[1:])(concat_attention)
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


class EntityEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(EntityEncoder, self).__init__()

        self.mha = MultiHeadAttention(d_model=464, num_heads=8)

    def call(self, embedded_feature_units):
        out, attn = self.mha(v=embedded_feature_units, k=embedded_feature_units, q=embedded_feature_units, mask=None)
        entity_embeddings = tf.keras.layers.Conv1D(256, 1, activation='relu')(out)

        embedded_entity = tf.reduce_mean(out, 1)
        embedded_entity = tf.cast(embedded_entity, tf.float32) 
        embedded_entity = tf.keras.layers.Dense(256, activation='relu')(embedded_entity)

        return embedded_entity, entity_embeddings


class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, img_height, img_width, channel):
    super(SpatialEncoder, self).__init__()

    self.map_model = tf.keras.Sequential([
       tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, channel)),
       tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(28, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
    ])

  def call(self, feature_screen):
    map_ = tf.transpose(feature_screen, perm=[0, 2, 3, 1])
    map_ = self.map_model(map_)
    map_ = tf.transpose(map_, perm=[0, 3, 1, 2])

    map_flatten = tf.keras.layers.Flatten()(map_)
    map_flatten = tf.cast(map_flatten, tf.float32) 

    embedded_spatial = tf.keras.layers.Dense(256, activation='relu')(map_flatten)

    return map_, embedded_spatial


class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number):
    super(Core, self).__init__()

    self.model = tf.keras.layers.LSTM(unit_number, return_sequences=True, return_state=True)

  def call(self, prev_state, embedded_entity, embedded_spatial, embedded_scalar):
    core_input = tf.concat((embedded_spatial, embedded_scalar, embedded_entity), axis=1)
    batch_size = tf.shape(core_input)[0]
    core_input = tf.keras.layers.Dense(128*2, activation='relu')(core_input)
    core_input = tf.reshape(core_input, (batch_size, -1, 128*2))
    lstm_output, final_memory_state, final_carry_state = self.model(core_input, initial_state=(prev_state[0], prev_state[1]), training=True)

    return lstm_output, final_memory_state, final_carry_state


def sample(a, temperature=0.8):
    return tf.argmax(a, axis=1)


class ResBlock_MLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.shortcut = tf.keras.layers.Dense(output_dim, activation='relu')

        self.mlp_0 = tf.keras.layers.Dense(output_dim, activation='relu')
        self.mlp_1 = tf.keras.layers.Dense(output_dim, activation='relu')

        self.bn_0 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        super(ResBlock_MLP, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.keras.layers.ReLU()(net)
        shortcut = self.shortcut(net)

        net = tf.cast(net, tf.float32) 
        net = self.mlp_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.mlp_1(net)

        output = net + shortcut
        return output


class ActionTypeHead(tf.keras.layers.Layer):
  def __init__(self, action_num):
    super(ActionTypeHead, self).__init__()

    self.action_num = action_num
    self.model = ResBlock_MLP(action_num)
  
  def call(self, lstm_output, scalar_context):
    #print("lstm_output.shape: " + str(lstm_output.shape))

    batch_size = tf.shape(scalar_context)[0]

    out = self.model(lstm_output, False)

    scalar_context = tf.cast(scalar_context, tf.float32) 
    out_gate = tf.keras.layers.Dense(self.action_num)(scalar_context)

    out_gated = out * out_gate

    action_type_logits = tf.keras.layers.Dense(self.action_num)(out_gated)
    action_type_logits = tf.reduce_mean(action_type_logits, axis=1)

    action_type = sample(action_type_logits)

    action_type_onehot = tf.one_hot(action_type, self.action_num)
    action_type_onehot = tf.cast(action_type_onehot, tf.float32) 

    autoregressive_embedding = tf.keras.layers.Dense(512)(action_type_onehot)
    autoregressive_embedding_gate = tf.keras.layers.Dense(512)(scalar_context)
    autoregressive_embedding = autoregressive_embedding * autoregressive_embedding_gate

    lstm_output_embedding = tf.cast(lstm_output, tf.float32) 
    lstm_output_embedding = tf.keras.layers.Dense(512)(lstm_output_embedding)
    lstm_output_embedding_gate = tf.keras.layers.Dense(512)(scalar_context)
    lstm_output_embedding = lstm_output_embedding * lstm_output_embedding_gate

    autoregressive_embedding += lstm_output_embedding

    return action_type_logits, action_type, autoregressive_embedding


class SelectedUnitsHead(tf.keras.layers.Layer):
  def __init__(self):
    super(SelectedUnitsHead, self).__init__()
    self.model = tf.keras.layers.Dense(256, activation='relu')

  def call(self, autoregressive_embedding, action_type, entity_embeddings):
    key = tf.keras.layers.Conv1D(512, 1, activation='relu')(entity_embeddings)

    autoregressive_embedding = tf.cast(autoregressive_embedding, tf.float32) 
    query = tf.keras.layers.Dense(512, activation='relu')(autoregressive_embedding)

    batch_size = tf.shape(entity_embeddings)[0]
    dim = tf.zeros([batch_size, 512])
    query, state_h, state_c = tf.keras.layers.LSTM(units=512, activation='relu', return_state=True, return_sequences=True)(query, 
                                                                                                                                                   initial_state=[dim, dim], 
                                                                                                                                                   training=True)
    selected_units_logits = tf.matmul(query, key, transpose_b=True)
    selected_units_logits = tf.reduce_mean(selected_units_logits, axis=1)
    selected_units = sample(selected_units_logits)

    selected_units_embedding = tf.one_hot(selected_units, 512)
    selected_units_embedding = tf.matmul(selected_units_embedding, key, transpose_b=True)
    selected_units_embedding = tf.cast(selected_units_embedding, tf.float32)

    autoregressive_embedding = tf.keras.layers.Dense(512, activation='relu')(autoregressive_embedding)
    autoregressive_embedding += selected_units_embedding

    return selected_units_logits, selected_units, autoregressive_embedding


class TargetUnitHead(tf.keras.layers.Layer):
  def __init__(self):
    super(TargetUnitHead, self).__init__()

    self.model = tf.keras.layers.Dense(256, activation='relu')

  def call(self, autoregressive_embedding, action_type, entity_embeddings):
    key = tf.keras.layers.Conv1D(32, 1, activation='relu')(entity_embeddings)

    autoregressive_embedding = tf.cast(autoregressive_embedding, tf.float32) 
    query = tf.keras.layers.Dense(256, activation='relu')(autoregressive_embedding)
    query = tf.keras.layers.Dense(32, activation='relu')(query)

    batch_size = tf.shape(autoregressive_embedding)[0]
    dim = tf.zeros([batch_size, 32])
    query, state_h, state_c = tf.keras.layers.LSTM(units=32, activation='relu', return_state=True, return_sequences=True)(query, 
                                                                                                                                                   initial_state=[dim, dim],
                                                                                                                                                   training=True)
    entity_selection_result = tf.matmul(query, key, transpose_b=True)

    target_unit_logits = entity_selection_result[0]
    target_unit = sample(target_unit_logits)

    return target_unit_logits, target_unit


class ResBlock_CNN(tf.keras.layers.Layer):
    def __init__(self, output_dim, strides=(1, 1, 1, 1), **kwargs):
        self.strides = strides
        if strides != (1, 1, 1, 1):
            self.shortcut = tf.keras.layers.Conv2D(4, 1, padding='same', activation='relu')

        self.conv_0 = tf.keras.layers.Conv2D(output_dim, 3, padding='same', activation='relu')
        self.conv_1 = tf.keras.layers.Conv2D(output_dim, 3, padding='same', activation='relu')
        self.bn_0 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        super(ResBlock_CNN, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)

        if self.strides != (1, 1, 1, 1):
            shortcut = self.shortcut(net)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = self.conv_1(net)

        output = net + shortcut
        return output


class LocationHead(tf.keras.layers.Layer):
  def __init__(self):
    super(LocationHead, self).__init__()

    self.model = ResBlock_CNN(32)

  def call(self, autoregressive_embedding, action_type, map_):
    batch_size = tf.shape(autoregressive_embedding)[0]

    autoregressive_embedding = tf.keras.layers.Dense(1024, activation='relu')(autoregressive_embedding)
    autoregressive_embedding_reshaped = tf.reshape(autoregressive_embedding, [batch_size, -1, 32, 32])
    map_concated = tf.concat((autoregressive_embedding_reshaped, map_), axis=1)

    target_location_logits = self.model(map_concated, True)

    target_location_logits = tf.keras.layers.Conv2DTranspose(5, 4, strides=2, padding='same', activation='relu', use_bias=False)(target_location_logits)
    target_location_logits = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='relu', use_bias=False)(target_location_logits)
    target_location_logits = tf.keras.layers.Dense(256, activation='relu')(target_location_logits)
    target_location_logits = tf.reshape(target_location_logits, [batch_size, -1, 256, 256])
    target_location_logits = tf.reduce_mean(target_location_logits, axis=1)

    target_location_logits_flatten = tf.keras.layers.Flatten()(target_location_logits)
    target_location = sample(target_location_logits_flatten)
    target_location = tf.cast(target_location, tf.int32) 

    x = tf.map_fn(lambda x: int(x / 256), target_location)
    y = tf.map_fn(lambda x: int(x % 256), target_location)

    target_location_logits = target_location_logits_flatten
    target_location = (x, y)

    return target_location_logits, target_location