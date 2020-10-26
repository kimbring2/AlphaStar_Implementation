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
    # output.shape: (1, 512, 32)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


class EntityEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(EntityEncoder, self).__init__()

        self.mha = MultiHeadAttention(d_model=464, num_heads=8)

    def call(self, embedded_feature_units):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        #y = tf.random.uniform((1, 512, 464))  # (batch_size, encoder_sequence, d_model)

        #embedded_feature_units = tf.cast(embedded_feature_units, tf.float32)
        #print("embedded_feature_units: " + str(embedded_feature_units))
        out, attn = self.mha(embedded_feature_units, k=embedded_feature_units, q=embedded_feature_units, mask=None)
        # out.shape: (1, 512, 32)
    
        '''
        The transformer output is passed through a ReLU, 1D convolution with 256 channels and kernel size 1, and another ReLU to yield 
        `entity_embeddings`. The mean of the transformer output across the units (masked by the missing entries) is fed through a linear 
        layer of size 256 and a ReLU to yield `embedded_entity`.
        '''

        entity_embeddings = tf.keras.layers.Conv1D(256, 1, activation='relu')(out)
        embedded_entity = tf.reduce_mean(out, 1)
        embedded_entity = tf.cast(embedded_entity, tf.float32) 
        embedded_entity = tf.keras.layers.Dense(256, activation='relu')(embedded_entity)
        #print("embedded_entity.shape: " + str(embedded_entity.shape))
        #embedded_entity.shape: (1, 464)

        return embedded_entity, entity_embeddings

'''
sample_decoder_layer = ScalarEncoder(464, 8)
sample_decoder_layer_output = sample_decoder_layer(tf.random.uniform((1,512)))
print("sample_decoder_layer_output.shape: " + str(sample_decoder_layer_output.shape))
'''

class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, img_height, img_width, channel):
    super(SpatialEncoder, self).__init__()

    self.map_model = tf.keras.Sequential([
       tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, channel)),
       tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D()
    ])

  def call(self, feature_screen):
    print("feature_screen.shape: " + str(feature_screen.shape))

    #feature_screen = feature_screen[0]
    '''
    Two additional map layers are added to those described in the interface. The first is a camera layer with two possible values: whether a location is inside 
    or outside the virtual camera. The second is the scattered entities. `entity_embeddings` are embedded through a size 32 1D convolution followed by a ReLU, 
    then scattered into a map layer so that the size 32 vector at a specific location corresponds to the units placed there. The planes are preprocessed as follows:
    '''

    '''
    After preprocessing, the planes are concatenated, projected to 32 channels by a 2D convolution with kernel size 1, passed through a ReLU, then downsampled 
    from 128x128 to 16x16 through 3 2D convolutions and ReLUs with channel size 64, 128, and 128 respectively. The kernel size for those 3 downsampling convolutions 
    is 4, and the stride is 2. 4 ResBlocks with 128 channels and kernel size 3 and applied to the downsampled map, with the skip connections placed into `map_skip`. 
    The ResBlock output is embedded into a 1D tensor of size 256 by a linear layer and a ReLU, which becomes `embedded_spatial`.
    '''
    #print("feature_screen.shape: " + str(feature_screen.shape))

    # out_entity.shape: (1, 510, 32)
    map_ = tf.transpose(feature_screen, perm=[0, 2, 3, 1])
    #map_ = tf.expand_dims(map_, axis=0)
    print("map_.shape: " + str(map_.shape))
    map_ = self.map_model(map_)

    map_flatten = tf.keras.layers.Flatten()(map_)
    # out_map.shape: (1, 16384)
    map_flatten = tf.cast(map_flatten, tf.float32) 

    embedded_spatial = tf.keras.layers.Dense(256, activation='relu')(map_flatten)
    #print("embedded_spatial.shape: " + str(embedded_spatial.shape))

    return map_, embedded_spatial

'''
sample_decoder_layer = SpatialEncoder(128, 128, 27)
sample_input = tf.random.uniform((27, 128, 128))
sample_input = tf.transpose(sample_input, perm=[1, 2, 0])
print("sample_input.shape: " + str(sample_input.shape))
sample_input = tf.reshape(sample_input, [1, 128, 128, 27])
print("sample_input.shape: " + str(sample_input.shape))

sample_decoder_layer_output = sample_decoder_layer([sample_input])

print("sample_decoder_layer_output.shape: " + str(sample_decoder_layer_output.shape))
'''

class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number):
    super(Core, self).__init__()

    self.model = tf.keras.layers.LSTM(unit_number, return_sequences=True, return_state=True)

  def call(self, prev_state, embedded_entity, embedded_spatial, embedded_scalar):
    #print("embedded_entity.shape: " + str(embedded_entity.shape))
    #print("embedded_scalar.shape: " + str(embedded_scalar.shape))
    #print("embedded_spatial.shape: " + str(embedded_spatial.shape))

    # enc_output.shape == (batch_size, input_seq_len, d_model)
    core_input = tf.concat((embedded_spatial, embedded_scalar, embedded_entity), axis=1)
    #print("encoder_input.shape: " + str(encoder_input.shape))
    # encoder_input.shape: (1, 819)

    core_input = tf.reshape(core_input, [1, 9, 91])
    lstm_output = self.model(core_input, initial_state=prev_state, training=True)
    #print("out_3.shape: " + str(out_3.shape))

    return lstm_output

'''
sample_core_layer = Core(12)

inputs = tf.random.normal([32, 10, 8])
whole_seq_output, final_memory_state, final_carry_state = sample_core_layer(inputs)
print(whole_seq_output.shape)
print(final_memory_state.shape)
print(final_carry_state.shape)
'''
'''
def sample(a, temperature=0.8):
  #print("a: " + str(a))

  a = np.array(a)**(1 / temperature)
  p_sum = a.sum()

  a = np.log(a) / temperature
  a = np.exp(a) / np.sum(np.exp(a))

  print("p_sum: " + str(p_sum))

  sample_temp = a / p_sum 

  #print("sample_temp.shape: " + str(sample_temp.shape))
  sample_temp = np.random.multinomial(1, sample_temp, 1)
  #print("sample_temp: " + str(sample_temp))

  return np.argmax(sample_temp)
'''

def sample(a, temperature=0.8):
    '''
    a = a + 1
    a = tf.math.log(a) / temperature
    a = tf.math.exp(a) / tf.reduce_sum(tf.exp(a))
    print("a: " + str(a))

    a_ = tf.compat.v1.distributions.Multinomial(total_count=1, logits=None, probs=1)
    return tf.argmax(a_)
    '''
    #a = a + 1
    #a = np.log(a) / temperature
    #a = np.exp(a) / np.sum(np.exp(a))
    #print("a: " + str(a))
    #print("tf.argmax(a): " + str(tf.argmax(a)))
    return tf.argmax(a)


class ResBlock_MLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        #self.shortcut = tf.keras.layers.Conv2D(output_dim, 1, strides=(1, 1), padding='same', activation='relu')
        self.shortcut = tf.keras.layers.Dense(256, activation='relu')

        #self.conv_0 = tf.keras.layers.Conv2D(output_dim, 3, padding='same', activation='relu')
        #self.conv_1 = tf.keras.layers.Conv2D(output_dim, 3, padding='same', activation='relu')
        self.mlp_0 = tf.keras.layers.Dense(256, activation='relu')
        self.mlp_1 = tf.keras.layers.Dense(256, activation='relu')

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
    self.model = ResBlock_MLP(256)
  
  def call(self, lstm_output,  scalar_context):
    out = self.model(lstm_output, True)
    scalar_context = tf.cast(scalar_context, tf.float32) 
    out_gate = tf.keras.layers.Dense(256)(scalar_context)
    #print("out_3.shape: " + str(out_3.shape))

    out_gated = tf.keras.layers.Multiply()([out, out_gate])
    output_flatten = tf.keras.layers.Flatten()(out_gated)
    output_flatten = tf.cast(output_flatten, tf.float32) 
    output = tf.keras.layers.Dense(self.action_num)(output_flatten)
    #print("output: " + str(output))

    action_type_logits = tf.nn.softmax(output, axis=-1)[0]
    #print("action_type_logits.shape: " + str(action_type_logits.shape))

    #action_type = tf.argmax(action_type_logits, axis=0)
    action_type = sample(action_type_logits)
    #print("action_type: " + str(action_type))

    action_type_onehot = tf.one_hot(action_type, self.action_num)
    action_type_onehot = tf.expand_dims(action_type_onehot, axis=0)
    #print("action_type_onehot.shape: " + str(action_type_onehot.shape))

    action_type_onehot = tf.cast(action_type_onehot, tf.float32) 
    autoregressive_embedding = tf.keras.layers.Dense(1024)(action_type_onehot)
    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))

    scalar_context = tf.cast(scalar_context, tf.float32) 
    autoregressive_embedding_gate = tf.keras.layers.Dense(1024)(scalar_context)
    autoregressive_embedding = tf.keras.layers.Multiply()([autoregressive_embedding, autoregressive_embedding_gate])
    # autoregressive_embedding.shape: (1, 1024)

    lstm_output_embedding = tf.keras.layers.Flatten()(lstm_output)
    lstm_output = tf.cast(lstm_output, tf.float32) 
    lstm_output_embedding = tf.keras.layers.Dense(1024)(lstm_output_embedding)
    lstm_output_embedding_gate = tf.keras.layers.Dense(1024)(scalar_context)
    lstm_output_embedding = tf.keras.layers.Multiply()([lstm_output_embedding, lstm_output_embedding_gate])
    #print("lstm_output_embedding.shape: " + str(lstm_output_embedding.shape))

    autoregressive_embedding = autoregressive_embedding + lstm_output_embedding
    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))
    '''
    `autoregressive_embedding` is then generated by first applying a ReLU and linear layer of size 256 to the one-hot version of `action_type`, 
     and projecting it to a 1D tensor of size 1024 through a `GLU` gated by `scalar_context`. That projection is added to another projection of 
     `lstm_output` into a 1D tensor of size 1024 gated by `scalar_context` to yield `autoregressive_embedding`.
    '''

    return action_type_logits, action_type, autoregressive_embedding


class SelectedUnitsHead(tf.keras.layers.Layer):
  def __init__(self):
    super(SelectedUnitsHead, self).__init__()

    self.model = tf.keras.layers.Dense(256, activation='relu')

  def call(self, autoregressive_embedding, action_acceptable_entity_type_binary, entity_embeddings):
    #action_acceptable_entity_type_onehot = tf.one_hot(action_acceptable_entity_type, 512)
    #action_acceptable_entity_type_binary = tf.expand_dims(action_acceptable_entity_type_binary, axis=0)

    #print("action_acceptable_entity_type_binary.shape: " + str(action_acceptable_entity_type_binary.shape))
    action_acceptable_entity_type_binary = tf.cast(action_acceptable_entity_type_binary, tf.float32) 
    func_embed = tf.keras.layers.Dense(256, activation='relu')(action_acceptable_entity_type_binary)

    '''
    It then computes a key corresponding to each entity by feeding `entity_embeddings` through a 1D convolution with 32 channels and kernel size 1, 
    and creates a new variable corresponding to ending unit selection. 
    '''

    #print("entity_embeddings.shape: " + str(entity_embeddings.shape))
    #entity_embeddings.shape: (1, 512, 256)
    key = tf.keras.layers.Conv1D(32, 1, activation='relu')(entity_embeddings)
    #print("key.shape: " + str(key.shape))
    # key.shape: (1, 512, 32) 
    #print("key: " + str(key))

    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))
    # autoregressive_embedding.shape: (1, 1024)
    autoregressive_embedding = tf.cast(autoregressive_embedding, tf.float32) 
    func_embed = tf.cast(func_embed, tf.float32) 
    query = tf.keras.layers.Dense(256, activation='relu')(autoregressive_embedding)
    query = tf.keras.layers.Dense(32, activation='relu')(func_embed + query)
    query = tf.expand_dims(query, axis=1)
    #print("query.shape: " + str(query.shape))

    batch_size = 1
    dim = tf.zeros([batch_size, 32])
    query, state_h, state_c = tf.keras.layers.LSTM(units=32, activation='relu', return_state=True, return_sequences=True)(query, 
                                                                                                                                                   initial_state=[dim, dim], 
                                                                                                                                                   training=True)
    #print("query.shape: " + str(query.shape))
    # query.shape: (1, 1, 32)
    #print("query: " + str(query))

    entity_selection_result = tf.matmul(query, key, transpose_b=True)
    #print("entity_selection_result.shape: " + str(entity_selection_result.shape))
    # entity_selection_result.shape: (1, 512, 32)

    units_logits = entity_selection_result[0][0]
    #print("units_logits: " + str(units_logits))

    #print("entity_selection_result[0][0].shape: " + str(entity_selection_result[0][0].shape))
    units = sample(entity_selection_result[0][0])
    #print("units: " + str(units))
    '''
    Then, repeated for selecting up to 64 units, the network passes `autoregressive_embedding` through a linear of size 256, adds `func_embed`, 
    and passes the combination through a ReLU and a linear of size 32. The result is fed into a LSTM with size 32 and zero initial state to get a query. 
    The entity keys are multiplied by the query, and are sampled using the mask and temperature 0.8 to decide which entity to select. That entity is 
    masked out so that it cannot be selected in future iterations. The one-hot position of the selected entity is multiplied by the keys, reduced by 
    the mean across the entities, passed through a linear layer of size 1024, and added to `autoregressive_embedding` for subsequent iterations. The 
    final `autoregressive_embedding` is returned. If `action_type` does not involve selecting units, this head is ignored.
    '''

    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))
    # autoregressive_embedding.shape: (1, 1024)

    autoregressive_embedding_ = tf.one_hot(units, 512)
    # autoregressive_embedding_.shape: (512,)
    # key.shape: (1, 512, 32)
    autoregressive_embedding_ = tf.keras.layers.Multiply()([autoregressive_embedding_, key[0]])
    autoregressive_embedding_ = tf.reduce_mean(autoregressive_embedding_, 0)
    #print("autoregressive_embedding_.shape: " + str(autoregressive_embedding_.shape))
    autoregressive_embedding_ = tf.expand_dims(autoregressive_embedding_, axis=0)

    autoregressive_embedding_ = tf.cast(autoregressive_embedding_, tf.float32) 
    autoregressive_embedding_ = tf.keras.layers.Dense(1024, activation='relu')(autoregressive_embedding_)
    autoregressive_embedding += autoregressive_embedding_
    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))

    return units_logits, units, autoregressive_embedding


class TargetUnitHead(tf.keras.layers.Layer):
  def __init__(self):
    super(TargetUnitHead, self).__init__()

    self.model = tf.keras.layers.Dense(256, activation='relu')

  def call(self,autoregressive_embedding, action_acceptable_entity_type_binary, entity_embeddings):
    #action_acceptable_entity_type_binary = tf.expand_dims(action_acceptable_entity_type_binary, axis=0)

    action_acceptable_entity_type_binary = tf.cast(action_acceptable_entity_type_binary, tf.float32) 
    func_embed = tf.keras.layers.Dense(256, activation='relu')(action_acceptable_entity_type_binary)

    key = tf.keras.layers.Conv1D(32, 1, activation='relu')(entity_embeddings)

    autoregressive_embedding = tf.cast(autoregressive_embedding, tf.float32) 
    query = tf.keras.layers.Dense(256, activation='relu')(autoregressive_embedding)
    query = tf.keras.layers.Dense(32, activation='relu')(func_embed + query)
    query = tf.expand_dims(query, axis=1)

    batch_size = 1
    dim = tf.zeros([batch_size, 32])
    query, state_h, state_c = tf.keras.layers.LSTM(units=32, activation='relu', return_state=True, return_sequences=True)(query, initial_state=[dim, dim], training=True)
    entity_selection_result = tf.matmul(query, key, transpose_b=True)

    target_unit_logits = entity_selection_result[0][0]
    target_unit = sample(entity_selection_result[0][0])

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

    self.model = ResBlock_CNN(20)

  def call(self, autoregressive_embedding, action_acceptable_entity_type, map_):
    '''
    `autoregressive_embedding` is reshaped to have the same height/width as the final skip in `map_skip` (which was just before map information was 
    reshaped to a 1D embedding) with 4 channels, and the two are concatenated together along the channel dimension, passed through a ReLU, passed 
    through a 2D convolution with 128 channels and kernel size 1, then passed through another ReLU. The 3D tensor (height, width, and channels) is 
    then passed through a series of Gated ResBlocks with 128 channels, kernel size 3, and FiLM, gated on `autoregressive_embedding` and using the 
    elements of `map_skip` in order of last ResBlock skip to first. Afterwards, it is upsampled 2x by each of a series of transposed 2D convolutions 
    with kernel size 4 and channel sizes 128, 64, 16, and 1 respectively (upsampled beyond the 128x128 input to 256x256 target location selection). 
    Those final logits are flattened and sampled (masking out invalid locations using `action_type`, such as those outside the camera for build 
    actions) with temperature 0.8 to get the actual target position. 
    '''
    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))
    # autoregressive_embedding.shape: (1, 1024)

    autoregressive_embedding_reshaped = tf.reshape(autoregressive_embedding, [16, 16, 4])
    autoregressive_embedding_reshaped = tf.expand_dims(autoregressive_embedding_reshaped, axis=0)
    #print("autoregressive_embedding_reshaped.shape: " + str(autoregressive_embedding_reshaped.shape))
    #print("map_.shape: " + str(map_.shape))
    map_concated = tf.concat((autoregressive_embedding_reshaped, map_), axis=3)
    #print("map_concated.shape: " + str(map_concated.shape))
    # map_concated.shape: (1, 16, 16, 20)

    target_location_logits = self.model(map_concated, True)
    #print("target_location_logits.shape: " + str(target_location_logits.shape))
    # target_location_logits.shape: (1, 16, 16, 20)
    target_location_logits = tf.keras.layers.Conv2DTranspose(10, 4, strides=2, padding='same', activation='relu', use_bias=False)(target_location_logits)
    #print("target_location_logits.shape: " + str(target_location_logits.shape))
    # target_location_logits.shape: (1, 32, 32, 10)

    target_location_logits = tf.keras.layers.Conv2DTranspose(5, 4, strides=2, padding='same', activation='relu', use_bias=False)(target_location_logits)
    #print("target_location_logits.shape: " + str(target_location_logits.shape))
    # target_location_logits.shape: (1, 64, 64, 5)

    target_location_logits = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='relu', use_bias=False)(target_location_logits)
    #print("target_location_logits.shape: " + str(target_location_logits.shape))
    # target_location_logits.shape: (1, 128, 128, 1)

    target_location_logits = tf.reshape(target_location_logits, [1, 128, 128])
    #print("target_location_logits.shape: " + str(target_location_logits.shape))
    # target_location_logits.shape: (1, 128, 128)

    target_location_logits_flatten = tf.keras.layers.Flatten()(target_location_logits)
    #print("target_location_logits_flatten[0].shape: " + str(target_location_logits_flatten[0].shape))
    #print("target_location_logits_flatten[0]: " + str(target_location_logits_flatten[0]))
    target_location = sample(target_location_logits_flatten[0])
    #print("target_location: " + str(target_location))

    x = int(target_location / 128)
    y = target_location % 128

    target_location_logits = target_location_logits_flatten[0]
    target_location = (x, y)
    #print("target_location: " + str(target_location))

    return target_location_logits, target_location