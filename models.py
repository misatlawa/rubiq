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

    with tf.variable_scope('Model'):
      self.q_predictions = self._q_predictions(self.states)
      self.action_predictions = tf.argmax(self.q_predictions, axis=1)

      self.q_next_state = self._q_predictions(self.next_states, reuse=True)
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
      name='Next_States'
    )

  def _mask(self):
    return tf.placeholder(
      shape=(None,),
      dtype=self.config.dtype,
      name='_mask'
    )

  def _q_predictions(self, input_, reuse=tf.AUTO_REUSE):
    for i, output_size in enumerate(self.config.hidden_layers):
      input_ = tf.layers.dense(
        input_,
        units=output_size,
        activation=tf.nn.tanh,
        name='Layer{}'.format(i),
        reuse=reuse,
      )
      tf.summary.histogram(input_.name, input_)

    return tf.layers.dense(
      input_,
      units=12,
      activation=tf.nn.tanh,
      use_bias=False,
      reuse=reuse
    )

  def _q_estimations(self):
    return tf.add(self.rewards,  self.gamma * self.next_state_value, name='q_estimations')

  def _loss(self):
    q_predictions = tf.reduce_sum(
      self.q_predictions * tf.one_hot(self.actions, self.config.action_size),
      axis=1
    )
    q_estimations = self._q_estimations()

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

  def act(self, state):
    action = self.session.run(
      fetches=(self.action_predictions),
      feed_dict={self.states: [state]}
    )
    return action[0]

  def save_weights(self, path_):
    self.saver.save(self.session, save_path=path_)

  def load_weights(self, path_):
    if path.isdir:
      path_ = tf.train.latest_checkpoint(path_)
    if path_:
      self.saver.restore(self.session, save_path=path_)
      print("restored weights from {}".format(path))

class DoubleDQN(Sequential):
  def __init__(self, config):
    self.config = config
    self.gamma = config.gamma

    with tf.name_scope('Batch'):
      self.states = self._states()
      self.actions = self._actions()
      self.rewards = self._rewards()
      self.next_states = self._next_states()

      self.mask = self._mask()

    with tf.variable_scope('Model') as scope:
      self.q_predictions = self._q_predictions(self.states)
      self.action_predictions = tf.argmax(self.q_predictions, axis=1)
      self.q_next_state = self._q_predictions(self.next_states, reuse=True)

      self._model_variables = scope.trainable_variables()

    with tf.variable_scope("Target") as scope:
      self.targets = self._q_predictions(self.next_states, reuse=False)
      self.next_state_value = tf.reduce_sum(
        self.targets * tf.one_hot(self.action_predictions, self.config.action_size),
        axis=1
      )
      self._target_variables = scope.trainable_variables()

    with tf.name_scope('Optimization'):
      self.global_step = tf.Variable(
        initial_value=0,
        dtype=tf.int64,
        name="step"
      )
      self.loss = self._loss()
      self.train_op = self._train_op()
      self.update_target = [x.assign(y) for x, y in zip(self._target_variables, self._model_variables)]

    self.session = tf.Session()
    self.saver = tf.train.Saver(max_to_keep=10)

    self.summary = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter(config.logdir, self.session.graph)
    self.init = tf.global_variables_initializer()
    self.session.run(self.init)

  def _train_op(self):
    optimizer = self.config.optimizer(self.config.learning_rate)
    return optimizer.minimize(
      self.loss,
      global_step=self.global_step,
      var_list=self._model_variables
    )

  def train(self, states, actions, rewards, next_states, mask_=None):
    train_result = super().train(states, actions, rewards, next_states, mask_)
    if train_result.step % self.config.update_interval == 0:
      self.session.run(
        self.update_target
      )
    return train_result
