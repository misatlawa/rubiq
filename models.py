from os import path, strerror
import errno

from attrdict import AttrDict
import numpy as np
import tensorflow as tf


class Sequential:
  def __init__(self, config):
    self.config = config
    self.gamma = config.gamma

    with tf.name_scope('Batch'):
      self.states = self._states()
      self.actions = self._actions()
      self.rewards = self._rewards()
      self.next_states = self._next_states()

      self.mask = self._mask()

    with tf.name_scope('Model'):
      self.q_predictions = self._q_predictions(self.states)
      self.action_prediction = tf.argmax(self.q_predictions, axis=1)

      self.q_next_state = self._q_predictions(self.next_states)
      self.next_state_value = tf.reduce_max(self.q_next_state)

    with tf.name_scope('Optimization'):

      self.global_step = tf.Variable(
        initial_value=0,
        dtype=tf.int64,
        name="step"
      )
      self.loss = self._loss()
      self.train_op = self._train_op()

    self.session = tf.Session()
    self.saver = tf.train.Saver(max_to_keep=10)

    self.summary = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter(config.logdir, self.session.graph)
    self.init = tf.global_variables_initializer()
    self.session.run(self.init)

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

  def _rewards(self):
    return tf.placeholder(
      shape=(None,),
      dtype=self.config.dtype,
      name='Rewards'
    )

  def _next_states(self):
    return tf.placeholder(
      shape=(None, self.config.state_size),
      dtype=self.config.dtype,
      name='Next_states'
    )

  def _mask(self):
    return tf.placeholder(
      shape=(None,),
      dtype=self.config.dtype,
      name='_mask'
    )

  def _q_predictions(self, input_):
    for i, output_size in enumerate(self.config.hidden_layers):
      input_ = tf.layers.dense(
        input_,
        units=output_size,
        activation=tf.nn.tanh,
        name='Layer{}'.format(i),
        reuse=tf.AUTO_REUSE,
      )
      tf.summary.histogram(input_.name, input_)
    q_predictions = tf.layers.dense(
      input_,
      units=12,
      activation=tf.nn.tanh,
      use_bias=False,
      reuse=tf.AUTO_REUSE
    )
    tf.summary.histogram('q_predictions', q_predictions)
    return q_predictions

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
    q_predictions = tf.reduce_sum(
      tf.math.multiply(actions, self.q_predictions),
      axis=1,
      name='q_predictions',
    )
    q_estimations = tf.add(self.rewards,  self.gamma * self.next_state_value, name='q_estimations')
    error = tf.multiply((q_predictions - q_estimations), self.mask, name='error')
    loss = tf.reduce_mean(tf.square(error), name='MSE_loss')

    tf.summary.scalar('loss', loss)
    tf.summary.histogram('q_prediction', q_predictions)
    tf.summary.histogram('q_estimation', q_estimations)

    return loss

  def _train_op(self):
    optimizer = self.config.optimizer(self.config.learning_rate)
    return optimizer.minimize(
      self.loss,
      global_step=self.global_step
    )

  def train(self, states, actions, rewards, next_states, mask_=None):
    mask_ = mask_ if mask_ is not None else np.ones_like(rewards)
    loss, step, summary, _ = self.session.run(
      fetches=(self.loss, self.global_step, self.summary, self.train_op),
      feed_dict={
        self.states: states,
        self.actions: actions,
        self.rewards: rewards,
        self.next_states: next_states,
        self.mask: mask_
      }
    )
    self.writer.add_summary(summary, global_step=step)
    return AttrDict(locals())

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


class DoubleDQN(Sequential):
  def __init__(self, config):
    super().__init__(config)
    cpu_config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    self.cpu_session = tf.Session(config=cpu_config)
    self.cpu_session.run(self.init)

  def act(self, state):
    action = self.cpu_session.run(
      fetches=(self.action_prediction),
      feed_dict={self.states: [state]}
    )
    return action[0]

  def update_weights(self):
    trainable_variables = tf.trainable_variables()
    variable_values = self.session.run(
      trainable_variables
    )
    self.cpu_session.run(
      [
        variable.assign(value)
        for variable, value in zip(trainable_variables, variable_values)
      ]
    )

  def train(self, states, actions, values):
    loss, step, summary, _ = self.session.run(
      fetches=(self.loss, self.global_step, self.summary, self.train_op),
      feed_dict={
        self.states: states,
        self.actions: actions,
        self.values: values
      }
    )
    self.writer.add_summary(summary, global_step=step)
    if not step % self.config.weight_update_interval:
      self.update_weights()

    return AttrDict(locals())
