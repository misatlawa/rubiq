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
      self.value = (self.value * self.n + value) / (self.n + 1)
      self.n = self.n + 1

  def __str__(self):
    return str(round(self.value, 4))


class DQNAgent:
  def __init__(self, config):
    self.config = config
    self.environment = RubiksCubeEnvironment(environment_config)
    self.exploration_rate = config.max_exploration_rate
    self.model = DoubleDQN(model_config)
    self.environments = [
      RubiksCubeEnvironment(environment_config)
      for _ in tqdm(range(100))
    ]
    self.memory = deque(maxlen=config.memory_size)

  def remember(self, state, action, reward, next_state):
    self.memory.append((state, action, reward, next_state))

  def policy(self, state):
    if np.random.rand() < self.exploration_rate:
      return np.random.choice(range(self.config.action_size))
    return self.model.act(state)

  def play_episode(self, is_eval=False):
    self.environment = RubiksCubeEnvironment(environment_config)
    self.environment.scramble(sample_scramble())
    state = self.environment.encoded_state()
    counter = 0
    for counter in range(self.environment.max_steps):
      action = self.model.act_one(state) if is_eval else self.policy(state)
      reward, next_state, is_terminal = self.environment(action)
      self.remember(state, action, reward, next_state)
      state = next_state
      if is_terminal:
        return counter, True
    return counter, False

  def play_episodes(self, n=None):
    n = n or len(self.environments)
    for env, diff in zip(self.environments[:n], sample_scramble(n)):
      env.reset()
      env.scramble(diff)

    states = [env.encoded_state() for env in self.environments]
    results = np.zeros(shape=(n, self.environment.max_steps))
    for step in range(self.environment.max_steps):
      actions = self.model.act(states)
      random_mask = np.random.binomial(n=1, p=self.exploration_rate, size=actions.shape)
      random_actions = np.random.choice(range(self.config.action_size), size=actions.shape)
      actions = random_mask * actions + (np.ones_like(random_mask) - random_mask) * random_actions
      responses = [self.environment(action) for action in actions]

      for i, (state, action, response) in enumerate(zip(states, actions, responses)):
        reward, next_state, is_terminal = response
        if not np.any(results[i, :]):
          self.remember(state, action, reward, next_state)
          results[i, step:] = is_terminal

      rewards, next_states, are_terminal = zip(*responses)
      states = next_states

    successful = np.max(results, axis=1)
    return np.average(successful)

  def train_on_batch(self):
    if len(self.memory) < self.model.config.batch_size:
      return
    batch = sample(self.memory, self.model.config.batch_size)
    states, actions, rewards, next_states = map(np.array, zip(*batch))

    return self.model.train(states, actions, rewards, next_states)

  def evaluation(self, n=None):
    n = n or agent.model.config.update_interval
    avg_length = Avg()
    avg_success = Avg()
    for _ in tqdm(range(n)):
      l, s = agent.play_episode(is_eval=True)
      avg_success.update(s)
      if s:
        avg_length.update(l)

    return avg_length, avg_success


if __name__ == "__main__":
  agent = DQNAgent(agent_config)
  avg_success = Avg()

  for _ in range(1000000):
    success_rate = agent.play_episodes(100)
    train_result = agent.train_on_batch()
    if train_result:
      if train_result.step % 1 == 0:
        print(
          "step: {}, loss: {}, success rate:{}".format(
            train_result.step,
            train_result and train_result.loss,
            success_rate)
        )
      if train_result.step %agent.model.config.update_interval == 0:
        avg_length, avg_success = agent.evaluation()
        summary = tf.Summary(value=[
          tf.Summary.Value(tag="success_rate", simple_value=avg_success.value),
          tf.Summary.Value(tag="avg_length", simple_value=avg_length.value),
          tf.Summary.Value(tag="exploration_rate", simple_value=agent.exploration_rate),
        ])
        agent.model.writer.add_summary(summary, global_step=train_result.step)
        agent.model.writer.flush()
        agent.exploration_rate = max(agent.config.min_exploration_rate, agent.exploration_rate * 0.99)

      if train_result.step % 3000 == 0:
        agent.model.save_weights('{}/s{}_{}'.format(
          agent.model.config.logdir,
          train_result.step,
          round(avg_success.value), 3)
        )