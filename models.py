from os import path

from attrdict import AttrDict
import numpy as np
import tensorflow as tf


class Sequential:
  def __init__(self, config):
    self.state_size = config.state_size
    self.action_size = config.action_size
    self.hidden_layers_size = config.hidden_layers_size
    self.dtype = config.dtype

    self._optimizer = config.optimizer
    self.learning_rate = config.learning_rate
    self.update_interval = config.update_interval
    self.gamma = config.gamma
    self.logdir = config.logdir

    with tf.name_scope('Input'):
      self.input = self._get_input()

    with tf.variable_scope('Network'):
      self.nn = self._get_network()

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
    self.writer = tf.summary.FileWriter(self.logdir, self.session.graph)
    self.init = tf.global_variables_initializer()
    self.session.run(self.init)

  def _states(self):
    return tf.placeholder(
      shape=(None, self.state_size),
      dtype=self.dtype,
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
      dtype=self.dtype,
      name='Rewards'
    )

  def _next_states(self):
    return tf.placeholder(
      shape=(None, self.state_size),
      dtype=self.dtype,
      name='Next_States'
    )

  def _mask(self):
    return tf.placeholder(
      shape=(None,),
      dtype=self.dtype,
      name='_mask'
    )

  def _get_input(self):
    states = self._states()
    actions = self._actions()
    rewards = self._rewards()
    next_states = self._next_states()
    mask = self._mask()
    return AttrDict(locals())

  def _get_network(self):
    q_predictions = self._q_predictions(input_=self.input.states)
    action_predictions = tf.argmax(q_predictions, axis=1)

    q_next_state = self._q_predictions(input_=self.input.next_states, reuse=True)
    next_state_value = tf.reduce_max(q_next_state)
    return AttrDict(locals())

  def _q_predictions(self, input_, reuse=tf.AUTO_REUSE):
    for i, output_size in enumerate(self.hidden_layers_size):
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
    return tf.add(self.input.rewards,  self.gamma * self.nn.next_state_value, name='q_estimations')

  def _loss(self):
    q_predictions = tf.reduce_sum(
      self.nn.q_predictions * tf.one_hot(self.input.actions, self.action_size),
      axis=1
    )
    q_estimations = self._q_estimations()

    error = tf.multiply((q_predictions - q_estimations), self.input.mask, name='error')
    loss = tf.reduce_mean(tf.square(error), name='MSE_loss')

    tf.summary.scalar('loss', loss)
    tf.summary.histogram('q_prediction', q_predictions)
    tf.summary.histogram('q_estimation', q_estimations)

    return loss

  def _train_op(self):
    optimizer = self._optimizer(self.learning_rate)
    return optimizer.minimize(
      self.loss,
      global_step=self.global_step
    )

  def train(self, states, actions, rewards, next_states, mask_=None):
    mask_ = mask_ if mask_ is not None else np.ones_like(rewards)

    loss, step, summary, _ = self.session.run(
      fetches=(self.loss, self.global_step, self.summary, self.train_op),
      feed_dict={
        self.input.states: states,
        self.input.actions: actions,
        self.input.rewards: rewards,
        self.input.next_states: next_states,
        self.input.mask: mask_
      }
    )
    self.writer.add_summary(summary, global_step=step)
    return AttrDict(locals())

  def act(self, state):
    action = self.session.run(
      fetches=(self.nn.action_predictions),
      feed_dict={self.input.states: [state]}
    )
    return action[0]

  def save_weights(self, path_):
    self.saver.save(self.session, save_path=path_)

  def load_weights(self, path_=None):
    path_ = path_ or self.logdir
    if path.isdir:
      path_ = tf.train.latest_checkpoint(path_)
    if path_:
      self.saver.restore(self.session, save_path=path_)
      print("Loaded model weights from {}".format(path_))


class DoubleDQN(Sequential):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    with tf.name_scope('Optimization'):
      self._update_target = [
        x.assign(y)
        for x, y in zip(self.nn.target_variables, self.nn.model_variables)
      ]

  def _get_network(self):
    with tf.variable_scope('Model') as scope:
      q_predictions = self._q_predictions(self.input.states)
      action_predictions = tf.argmax(q_predictions, axis=1)
      q_next_state = self._q_predictions(self.input.next_states, reuse=True)
      model_variables = scope.trainable_variables()

    with tf.variable_scope("Target") as scope:
      targets = self._q_predictions(self.input.next_states, reuse=False)
      next_state_value = tf.reduce_sum(
        targets * tf.one_hot(action_predictions, self.action_size),
        axis=1
      )
      target_variables = scope.trainable_variables()

    return AttrDict(locals())

  def _train_op(self):
    optimizer = self._optimizer(self.learning_rate)
    return optimizer.minimize(
      self.loss,
      global_step=self.global_step,
      var_list=self.nn.model_variables
    )

  def train(self, states, actions, rewards, next_states, mask_=None):
    train_result = super().train(states, actions, rewards, next_states, mask_)
    if train_result.step % self.update_interval == 0:
      self.session.run(
        self._update_target
      )
    return train_result
