import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow_probability.python.distributions import kullback_leibler

tfd = tfp.distributions


class ScalarEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ScalarEncoder, self).__init__()
    self.output_dim = output_dim

    self.network = tf.keras.Sequential([
       tf.keras.layers.Dense(self.output_dim, activation='relu', name="ScalarEncoder_dense")
    ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, scalar_feature):
    batch_size = tf.shape(scalar_feature)[0]
    scalar_feature_encoded = self.network(scalar_feature)
    return scalar_feature_encoded


class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, height, width, channel):
    super(SpatialEncoder, self).__init__()

    self.height = height
    self.width = width

    self.network = tf.keras.Sequential([
       tf.keras.layers.Conv2D(13, 1, padding='same', activation='relu', name="SpatialEncoder_cond2d_1"),
       tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu', name="SpatialEncoder_cond2d_2"),
       tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name="SpatialEncoder_cond2d_3")
    ])

    self.dense = tf.keras.layers.Dense(256, activation='relu', name="SpatialEncoder_dense")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
    })
    return config

  def call(self, spatial_feature):
    spatial_feature_encoded = self.network(spatial_feature)

    return spatial_feature_encoded


class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number):
    super(Core, self).__init__()

    self.unit_number = unit_number
    self.network = tf.keras.layers.Dense(256, activation='relu', name="core_dense")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'unit_number': self.unit_number
    })
    return config

  def call(self, feature_encoded):
    batch_size = tf.shape(feature_encoded)[0]

    feature_encoded_flattened = Flatten()(feature_encoded)

    core_output = self.network(feature_encoded_flattened)

    return core_output


class Baseline(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(Baseline, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.layers.Dense(1, activation='relu')

    self.autoregressive_embedding_encoder = tf.keras.layers.Dense(self.output_dim, activation='relu', 
                                                                                 name="Baseline_dense")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, feature_encoded, core_output):
    batch_size = tf.shape(core_output)[0]

    feature_encoded_flattened = tf.keras.layers.Flatten()(feature_encoded)
    feature_encoded_flattened_embedding = self.autoregressive_embedding_encoder(feature_encoded_flattened)

    network_input = tf.concat([core_output, feature_encoded_flattened_embedding], axis=1)
    value = self.network(network_input)

    return value


class ActionTypeHead(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ActionTypeHead, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.layers.Dense(self.output_dim, activation='softmax', name="ActionTypeHead_dense_1")
    self.autoregressive_embedding_network = tf.keras.layers.Dense(256, activation='relu', name="ActionTypeHead_dense_2")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, core_output):
    batch_size = tf.shape(core_output)[0]
    action_type_logits = self.network(core_output)

    action_type_dist = tfd.Categorical(probs=action_type_logits)
    action_type = action_type_dist.sample()
    action_type_onehot = tf.one_hot(action_type, self.output_dim)

    autoregressive_embedding = self.autoregressive_embedding_network(action_type_onehot)
    autoregressive_embedding += core_output

    return action_type_logits, autoregressive_embedding


class SpatialArgumentHead(tf.keras.layers.Layer):
  def __init__(self, height, width, channel):
    super(SpatialArgumentHead, self).__init__()

    self.height = height
    self.width = width

    self.network = tf.keras.Sequential()
    self.network.add(tf.keras.layers.Conv2D(1, 1, padding='same', name="SpatialArgumentHead_conv2d"))
    self.network.add(tf.keras.layers.Flatten())
    self.network.add(tf.keras.layers.Softmax())

    self.autoregressive_embedding_encoder_1 = tf.keras.layers.Dense(self.height * self.width, activation='relu', 
                                                                                  name="SpatialArgumentHead_dense_1")
    self.autoregressive_embedding_encoder_2 = tf.keras.layers.Dense(self.height * self.width, activation='relu', 
                                                                                  name="SpatialArgumentHead_dense_2")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
    })
    return config

  def call(self, feature_encoded, core_output, autoregressive_embedding):
    batch_size = tf.shape(core_output)[0]

    encoded_core_output = self.autoregressive_embedding_encoder_1(core_output)
    encoded_core_output = tf.reshape(encoded_core_output, (batch_size, self.height, self.width, 1))

    encoded_autoregressive_embedding = self.autoregressive_embedding_encoder_2(autoregressive_embedding)
    encoded_autoregressive_embedding = tf.reshape(encoded_autoregressive_embedding, (batch_size, self.height, self.width, 1))

    network_input = tf.concat([feature_encoded, encoded_core_output, encoded_autoregressive_embedding], axis=3)
    action_logits = self.network(network_input)

    return action_logits


class ScalarArgumentHead(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ScalarArgumentHead, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.Sequential()
    self.network.add(tf.keras.layers.Dense(output_dim, name="ScalarArgumentHead_dense_1"))
    self.network.add(tf.keras.layers.Softmax())

    self.feature_encoded_encoder = tf.keras.layers.Dense(self.output_dim, activation='relu', 
                                                                     name="ScalarArgumentHead_dense_1")
    self.autoregressive_embedding_encoder = tf.keras.layers.Dense(self.output_dim, activation='relu', 
                                                                                 name="ScalarArgumentHead_dense_2")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, core_output, autoregressive_embedding):
    batch_size = tf.shape(core_output)[0]

    #feature_encoded_flattened = tf.keras.layers.Flatten()(feature_encoded)
    #feature_encoded_flattened_embedding = self.feature_encoded_encoder(feature_encoded_flattened)

    encoded_autoregressive_embedding = self.autoregressive_embedding_encoder(autoregressive_embedding)

    network_input = tf.concat([core_output, encoded_autoregressive_embedding], axis=1)
    action_logits = self.network(network_input)
    
    return action_logits