from os import path, strerror
import errno

import tensorflow as tf


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
      self.state_value_prediction = tf.reduce_max(self.q_predictions, axis=1)

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
    #self.cpu_session = tf.Session(config=cpu_config)
    self.session = tf.Session()
    self.saver = tf.train.Saver()
    self.init = tf.global_variables_initializer()
    self.session.run(self.init)
    #self.cpu_session.run(self.init)

  def _states(self):
    return tf.placeholder(
      shape=(None, self.config.state_size),
      dtype=self.config.dtype,
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
      shape=(None,),
      dtype=self.config.dtype,
      name='Values'
    )

  def _q_predictions(self):
    input_ = self.states
    for output_size in self.config.hidden_layers:
      input_ = tf.contrib.layers.fully_connected(
        inputs=input_,
        num_outputs=output_size,
        activation_fn=tf.nn.sigmoid
      )

    return tf.contrib.layers.fully_connected(
      inputs=input_,
      num_outputs=self.config.action_size,
      activation_fn=tf.nn.sigmoid,
    )

  def act(self, state):
    action = self.session.run(
      fetches=(self.action_prediction),
      feed_dict={self.states: [state]}
    )
    return action[0]

  def _loss(self):
    actions = tf.one_hot(
      self.actions,
      self.config.action_size
    )
    q_actions = tf.reduce_max(
      tf.math.multiply(actions, self.q_predictions),
      axis=1
    )
    return tf.reduce_mean(
      tf.square(q_actions - self.values)
    )

  def _train_op(self):
    optimizer = self.config.optimizer(self.config.learning_rate)
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
        strerror(errno.ENOENT),
        path_
      )
