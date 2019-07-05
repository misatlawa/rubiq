from os import path


import tensorflow as tf

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


class Sequential:
  def __init__(self, config):
    self.config = config

    with tf.name_scope('Model'):
      self.input = self._input()
      self.q_values = self._q_values()
      self.q_predictions = self._q_predictions()
      self._action = tf.argmax(self.q_predictions, axis=1)

      self.global_step = tf.Variable(
        initial_value=0,
        dtype=tf.int32,
        name="step"
      )
      self.loss = self._loss()
      self.train_op = self._train_op()

    cpu_config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    self.cpu_session = tf.Session(config=cpu_config)
    self.session = tf.Session()
    self.saver = tf.train.Saver()
    self.init = tf.global_variables_initializer()
    self.session.run(self.init)
    self.cpu_session.run(self.init)

  def _input(self):
    return tf.placeholder(
      shape=(None, self.config.STATE_SIZE),
      dtype=self.config.DTYPE,
      name='Input'
    )

  def _q_values(self):
    return tf.placeholder(
      shape=(None, self.config.ACTION_SIZE),
      dtype=self.config.DTYPE,
      name='Q-values'
    )

  def _q_predictions(self):
    input_ = self.input
    input_size = self.config.STATE_SIZE
    for i, output_size in enumerate(self.config.LAYERS):
      weights = tf.get_variable(
        shape=(input_size, output_size),
        dtype=config.DTYPE,
        name='weight{0}'.format(i)
      )
      bias = tf.get_variable(
        shape=output_size,
        dtype=config.DTYPE,
        name='bias{0}'.format(i)
      )
      input_ = tf.nn.relu(tf.matmul(input_, weights) + bias)
      input_size = output_size
    return input_

  def action(self, state):
    return self.cpu_session.run(
      fetches=self._action,
      feed_dict={self.input: [state]}
    )

  def _loss(self):
    return tf.reduce_mean(
      tf.square(self.q_predictions - self.q_values)
    )

  def _train_op(self):
    optimizer = self.config.OPTIMIZER(self.config.LEARNING_RATE)
    return optimizer.minimize(
      self.loss,
      global_step=self.global_step
    )

  def train(self, states, q_values):
    loss, _ = self.session.run(
      fetches=(self.loss, self.train_op),
      feed_dict={
        self.input: states,
        self.q_values: q_values
      }
    )
    return loss

  def save_weights(self, path_):
    self.saver.save(self.session, save_path=path_)

  def load_weights(self, path_):
    if path.isdir:
      self.saver.recover_last_checkpoints(checkpoint_paths=path_)
    elif path.isfile:
      self.saver.restore(self.session, save_path=path_)
    else:
      raise FileNotFoundError

if __name__ == '__main__':
  import numpy as np

  config = AttrDict(
    {
      "STATE_SIZE": 20 * 24,
      "ACTION_SIZE": 6,
      "LAYERS": [100, 100, 100, 6],
      "DTYPE": tf.float16,
      "OPTIMIZER": tf.train.GradientDescentOptimizer,
      "LEARNING_RATE": 1e-3
    }
  )
  s = Sequential(config)
  for _ in range(3000):
    states = np.random.uniform(size=10*20*24)
    states = np.reshape(states, newshape=(10, 20*24))
    q_values = states[:, :6]
    print(s.train(states, q_values))
