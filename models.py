from os import path

import tensorflow as tf


class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


config = AttrDict(
  {
    "STATE_SIZE": 20 * 24,
    "ACTION_SIZE": 6,
    "HLAYERS": [100, 100, 100],
    "DTYPE": tf.float32,
    "OPTIMIZER": tf.train.GradientDescentOptimizer,
    "LEARNING_RATE": 1e-3,
  }
)


class Sequential:
  def __init__(self, config):
    self.config = config

    with tf.name_scope('Batch'):
      self.states = self._states()
      self.actions = self._actions()
      self.values = self._values()

    with tf.name_scope('Model'):
      self.q_predictions = self._q_predictions()
      self.action_prediction = tf.argmax(self.q_predictions, axis=1)

    with tf.name_scope('Optimization'):
      self.global_step = tf.Variable(
        initial_value=0,
        dtype=tf.int64,
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

  def _states(self):
    return tf.placeholder(
      shape=(None, self.config.STATE_SIZE),
      dtype=self.config.DTYPE,
      name='States'
    )

  def _actions(self):
    return tf.placeholder(
      shape=(None,),
      dtype=tf.int32,
      name='Actions'
    )

  def _values(self):
    return tf.placeholder(
      shape=(None, self.config.ACTION_SIZE),
      dtype=self.config.DTYPE,
      name='Values'
    )

  def _q_predictions(self):
    input_ = self.states
    for output_size in self.config.HLAYERS:
      input_ = tf.contrib.layers.fully_connected(
        inputs=input_,
        num_outputs=output_size,
        activation_fn=tf.nn.relu
      )

    return tf.contrib.layers.fully_connected(
      inputs=input_,
      num_outputs=self.config.ACTION_SIZE,
      activation_fn=None
    )

  def act(self, state):
    return self.cpu_session.run(
      fetches=self.action_prediction,
      feed_dict={self.input: [state]}
    )[0]

  def _loss(self):
    actions = tf.one_hot(
      self.actions,
      self.config.ACTION_SIZE
    )
    q_actions = tf.reduce_sum(tf.mul(actions, self.q_predictions), axis=1)

    return tf.reduce_mean(
      tf.square(q_actions - self.values)
    )

  def _train_op(self):
    optimizer = self.config.OPTIMIZER(self.config.LEARNING_RATE)
    return optimizer.minimize(
      self.loss,
      global_step=self.global_step
    )

  def train(self, states, actions, values):
    loss, _ = self.session.run(
      fetches=(self.loss, self.train_op),
      feed_dict={
        self.states: states,
        self.actions: actions,
        self.values: values
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
      raise FileNotFoundError(
        errno.ENOENT,
        os.strerror(errno.ENOENT),
        path_
      )
