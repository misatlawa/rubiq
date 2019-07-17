from collections import deque
import numpy as np
from random import sample
from tqdm import tqdm
import tensorflow as tf

from configs import model_config, agent_config, environment_config
from environment import RubiksCubeEnvironment
from models import DoubleDQN


def sample_scramble(size=None):
  return np.random.randint(low=1, high=30, size=size)


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

    self.environment = RubiksCubeEnvironment(environment_config)
    self.model = DoubleDQN(model_config)

  def remember(self, state, action, reward, next_state):
    self.memory.append((state, action, reward, next_state))

  def policy(self, state):
    if np.random.rand() < self.exploration_rate:
      return np.random.choice(range(self.action_size))
    return self.model.act(state)

  def play_episode(self, is_eval=False):
    self.environment = RubiksCubeEnvironment(environment_config)
    self.environment.scramble(sample_scramble())
    state = self.environment.encoded_state()
    is_terminal = False
    counter = 0
    while not is_terminal:
      action = self.model.act(state) if is_eval else self.policy(state)
      reward, next_state, is_terminal = self.environment(action)
      self.remember(state, action, reward, next_state)
      counter += 1
      state = next_state
    return counter, reward == self.environment.success_reward

  def train_on_batch(self, batch=None):
    if len(self.memory) < self.batch_size:
      return
    batch = batch or sample(self.memory, self.batch_size)
    states, actions, rewards, next_states = map(np.array, zip(*batch))

    return self.model.train(states, actions, rewards, next_states)

  def evaluation(self, n=None):
    n = n or agent.model.update_interval
    avg_length = Avg()
    avg_success = Avg()
    for _ in tqdm(range(n)):
      l, s = agent.play_episode(is_eval=True)
      avg_success.update(s)
      if s and l:
        avg_length.update(l)

    return avg_length, avg_success


if __name__ == "__main__":
  agent = DQNAgent(agent_config)
  agent.model.load_weights(agent.model.logdir)
  avg_success = Avg()

  for _ in range(1000000):
    l, s = agent.play_episode()
    train_result = agent.train_on_batch()
    if train_result:
      if train_result.step % 100 == 0:
        print(
          "step: {}, loss: {}".format(train_result.step, train_result and train_result.loss)
        )
      if train_result.step % agent.model.update_interval == 0:
        avg_length, avg_success = agent.evaluation()
        summary = tf.Summary(value=[
          tf.Summary.Value(tag="success_rate", simple_value=avg_success.value),
          tf.Summary.Value(tag="avg_length", simple_value=avg_length.value),
          tf.Summary.Value(tag="exploration_rate", simple_value=agent.exploration_rate),
        ])
        agent.model.writer.add_summary(summary, global_step=train_result.step)
        agent.model.writer.flush()
        agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * 0.9)

      if train_result.step % 3000 == 0:
        agent.model.save_weights('{}/s{}_{}'.format(
          agent.model.logdir,
          train_result.step,
          round(avg_success.value), 3)
        )