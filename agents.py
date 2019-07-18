from collections import deque
import numpy as np
from random import sample
from tqdm import tqdm
import tensorflow as tf

from configs import model_config, agent_config, environment_config
from environment import TensorEnvironment, RubiksCubeEnvironment
from models import DoubleDQN


def sample_scramble(size=None):
  return np.random.randint(low=1, high=20, size=size)


class Avg:
  def __init__(self, value=None, n=0):
    self.value = value
    self.n = n

  def update(self, value):
    if value is not None:
      if self.value is None:
        self.value = 0
      self.value = (self.value * self.n + value) / (self.n + 1 )
      self.n = self.n + 1

  def __str__(self):
    return str(round(self.value, 4))


class DQNAgent:
  def __init__(self, config):
    self.memory = deque(maxlen=config.memory_size)
    self.state_size = config.state_size
    self.action_size = config.action_size
    self.memory_size = config.memory_size
    self.batch_size = config.batch_size

    self.exploration_rate = config.max_exploration_rate
    self.min_exploration_rate = config.min_exploration_rate

    self.evaluation_environment = RubiksCubeEnvironment(environment_config)
    self.training_environment = TensorEnvironment(environment_config, n=self.batch_size)
    self.model = DoubleDQN(model_config)

  def remember(self, *args):
    self.memory.append(args)

  def policy(self, state):
    if np.random.rand() < self.exploration_rate:
      return np.random.choice(range(self.action_size))
    return self.model.act(state)

  def play_episode(self, difficulty=None):
    difficulty = difficulty or sample_scramble()
    self.evaluation_environment.reset()
    self.evaluation_environment.scramble(difficulty)

    state = self.evaluation_environment.encoded_state()
    is_terminal = False
    counter = 0
    while not is_terminal:
      action = self.model.act(state)
      reward, next_state, is_terminal = self.evaluation_environment(action)
      self.remember(state, action, reward, next_state, is_terminal)
      counter += 1
      state = next_state
    return reward == self.evaluation_environment.success_reward, counter

  def train_online(self, difficulty=30):
    self.training_environment.reset()
    self.training_environment.scramble(difficulty)
    states = self.training_environment.state

    active = np.ones(shape=self.batch_size)
    success = Avg()
    lengths = Avg()
    for i in range(1, 30):
      random_mask = np.random.binomial(1, p=self.exploration_rate, size=self.batch_size)
      actions = random_mask * self.model.act(states) + (1 - random_mask) * np.random.choice(range(self.action_size))
      response = self.training_environment(actions)
      rewards, next_states, terminal = response

      self.model.train(states, actions, *response, mask_=active)
      for state, action, reward, next_state, is_terminal, is_active in zip(states, actions, *response, active):
        if is_active:
          self.remember(state, action, reward, next_state, is_terminal)
          if is_terminal:
            if reward == self.training_environment.success_reward:
              success.update(1)
              lengths.update(i)
            else:
              success.update(0)

      active *= np.logical_not(terminal)
      if not np.any(terminal):
        break

    return success.value, lengths.value

  def train_on_memory(self, batch=None):
    if len(self.memory) < self.batch_size:
      return
    batch = batch or sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, is_terminal = map(np.array, zip(*batch))

    return self.model.train(states, actions, rewards, next_states, is_terminal)

  def evaluation(self, n=None):
    n = n or self.model.update_interval
    avg_length = Avg()
    avg_success = Avg()
    for _ in tqdm(range(n)):
      s, l = self.play_episode()
      avg_success.update(s)
      if s and l:
        avg_length.update(l)

    return avg_success, avg_length


if __name__ == '__main__':
  agent = DQNAgent(agent_config)
  agent.model.load_weights(agent.model.logdir)
  avg_success = Avg()

  for step in range(1000000):
    difficulty = sample_scramble()
    success = Avg()
    length = Avg()

    for _ in range(10):
      s, l = agent.play_episode(difficulty)
      success.update(s)
      if s:
        length.update(l)

    #success, length = agent.train_online(difficulty)
    train_result = agent.train_on_memory()
    if train_result:
      print(
        'step: {} ({}), loss: {}, acc: {} ({})'.format(train_result.step, step, train_result.loss, success, difficulty)
      )
      if difficulty < 10:
        summary = tf.Summary(value=[
          tf.Summary.Value(tag='n_success_rate/{}_success_rate'.format(difficulty), simple_value=success.value),
          tf.Summary.Value(tag='n_solution_length/{}_solution_length'.format(difficulty), simple_value=length.value),
        ])
        agent.model.writer.add_summary(summary, global_step=train_result.step)
      if step % 10 == 0:
        avg_success, avg_length = agent.evaluation()
        summary = tf.Summary(value=[
          tf.Summary.Value(tag='success_rate', simple_value=avg_success.value),
          tf.Summary.Value(tag='solution_length', simple_value=avg_length.value),
          tf.Summary.Value(tag='exploration_rate', simple_value=agent.exploration_rate),
        ])
        agent.model.writer.add_summary(summary, global_step=train_result.step)
        agent.model.writer.flush()
        agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * 0.9)

      if step % 1000 == 0:
        agent.model.save_weights('{}/s{}_{}'.format(
          agent.model.logdir,
          train_result.step,
          round(avg_success.value), 3)
        )
    step += 1