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
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


class EntityEncoder(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, rate=0.1):
		super(EntityEncoder, self).__init__()

		self.temp_mha = MultiHeadAttention(d_model=464, num_heads=8)
		#y = tf.random.uniform((1, 512, 464))  # (batch_size, encoder_sequence, d_model)
		#out, attn = temp_mha(y, k=y, q=y, mask=None)
		#print("out.shape: " + str(out.shape))
		#print("attn.shape: " + str(attn.shape))

		self.relu_layer = tf.keras.layers.ReLU()
		#out_1 = relu_layer(out)
		#print("out_1.shape: " + str(out_1.shape))

		#input_shape = out_1.shape
		#self.conv1d_layer = tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape[1:])
		#out_2 = conv1d_layer(out_1)
		#out_2 = tf.reshape(out_2, (1, 510 * 32))
		#print("out_2.shape: " + str(out_2.shape))

		self.linear_layer = tf.keras.layers.Dense(256, activation='relu')
		#out_3 = linear_layer(out_2)
		#print("out_3.shape: " + str(out_3.shape))

	def call(self, y):
	    # enc_output.shape == (batch_size, input_seq_len, d_model)

		#y = tf.random.uniform((1, 512, 464))  # (batch_size, encoder_sequence, d_model)
		out1, attn1 = self.temp_mha(y, k=y, q=y, mask=None)
		input_shape = out1.shape
		#print("out_1.shape: " + str(out_1.shape))

		out2 = tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape[1:])(out1)
		out2 = tf.reshape(out2, (1, 510 * 32))
		#print("out_2.shape: " + str(out_2.shape))

		out3 = 	self.linear_layer(out2)
		#print("out_3.shape: " + str(out_3.shape))

		return out3


sample_decoder_layer = EntityEncoder(464, 8)
sample_decoder_layer_output = sample_decoder_layer(tf.random.uniform((1, 512, 464)))

print("sample_decoder_layer_output.shape: " + str(sample_decoder_layer_output.shape))