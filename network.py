import tensorflow as tf


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

    def call(self, y):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        #y = tf.random.uniform((1, 512, 464))  # (batch_size, encoder_sequence, d_model)
        out, attn = self.mha(y, k=y, q=y, mask=None)
        # out.shape: (1, 512, 32)
    
        '''
        The transformer output is passed through a ReLU, 1D convolution with 256 channels and kernel size 1, and another ReLU to yield 
        `entity_embeddings`. The mean of the transformer output across the units (masked by the missing entries) is fed through a linear 
        layer of size 256 and a ReLU to yield `embedded_entity`.
        '''

        entity_embeddings = tf.keras.layers.Conv1D(256, 1, activation='relu')(out)
        embedded_entity = tf.reduce_mean(out, 1)
        embedded_entity = tf.keras.layers.Dense(256, activation='relu')(embedded_entity)
        #print("embedded_entity.shape: " + str(embedded_entity.shape))
        #embedded_entity.shape: (1, 464)

        return embedded_entity, entity_embeddings


class ScalarEncoder(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super(ScalarEncoder, self).__init__()

    self.model = tf.keras.layers.Dense(d_model, activation='relu')

  def call(self, x):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    out = self.model(x)
    #print("out_3.shape: " + str(out_3.shape))

    return out

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
       tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Flatten()
    ])

    self.entity_model = tf.keras.Sequential([
       tf.keras.layers.Conv1D(32, 3, activation='relu')
    ])

  def call(self, x, y):
      # enc_output.shape == (batch_size, input_seq_len, d_model)

    out = self.map_model(x)
    out_entity = self.entity_model(y)
    print("out_entity.shape: " + str(out_entity.shape))

    return out

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

  def call(self, x):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    out = self.model(x)
    #print("out_3.shape: " + str(out_3.shape))

    return out

'''
sample_core_layer = Core(12)

inputs = tf.random.normal([32, 10, 8])
whole_seq_output, final_memory_state, final_carry_state = sample_core_layer(inputs)
print(whole_seq_output.shape)
print(final_memory_state.shape)
print(final_carry_state.shape)
'''

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        #self.shortcut = tf.keras.layers.Conv2D(output_dim, 1, strides=(1, 1), padding='same', activation='relu')
        self.shortcut = tf.keras.layers.Dense(256, activation='relu')

        #self.conv_0 = tf.keras.layers.Conv2D(output_dim, 3, padding='same', activation='relu')
        #self.conv_1 = tf.keras.layers.Conv2D(output_dim, 3, padding='same', activation='relu')
        self.mlp_0 = tf.keras.layers.Dense(256, activation='relu')
        self.mlp_1 = tf.keras.layers.Dense(256, activation='relu')

        self.bn_0 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        super(ResBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.keras.layers.ReLU()(net)

        shortcut = self.shortcut(net)

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
    self.model = ResBlock(256)

  def call(self, x,  y):
    out = self.model(x, False)
    out_gate = tf.keras.layers.Dense(256)(y)
    #print("out_3.shape: " + str(out_3.shape))

    out_gated = tf.keras.layers.Multiply()([out, out_gate])
    output_flatten = tf.keras.layers.Flatten()(out_gated)
    output = tf.keras.layers.Dense(self.action_num)(output_flatten)
    action_type_logits = tf.nn.softmax(output, axis=-1)[0]
    #print("action_type_logits: " + str(action_type_logits))

    action_type = tf.argmax(action_type_logits)
    #print("action_type: " + str(action_type))

    action_type_onehot = tf.one_hot(action_type, self.action_num)
    action_type_onehot = tf.expand_dims(action_type_onehot, axis=0)
    #print("action_type_onehot.shape: " + str(action_type_onehot.shape))

    autoregressive_embedding = tf.keras.layers.Dense(1024)(action_type_onehot)
    #print("autoregressive_embedding.shape: " + str(autoregressive_embedding.shape))

    autoregressive_embedding_gate = tf.keras.layers.Dense(1024)(y)
    autoregressive_embedding = tf.keras.layers.Multiply()([autoregressive_embedding, autoregressive_embedding_gate])
    # autoregressive_embedding.shape: (1, 1024)

    lstm_output_embedding = tf.keras.layers.Flatten()(x)
    lstm_output_embedding = tf.keras.layers.Dense(1024)(lstm_output_embedding)
    lstm_output_embedding_gate = tf.keras.layers.Dense(1024)(y)
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
