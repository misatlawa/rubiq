import tensorflow as tf

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

STATE_SIZE = 20 * 24
ACTION_SIZE = 6

model_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": ACTION_SIZE,
    "hidden_layers": [100, 100, 100],
    "dtype": tf.float32,
    "optimizer": tf.train.GradientDescentOptimizer,
    "learning_rate": 1e-3,
  }
)

agent_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": ACTION_SIZE,
    "memory_size": 10000,
    "exploration_rate": 1,
    "gamma": 0.9
  }
)

environment_config = AttrDict(
  {
    "step_reward": -1,
    "success_reward": 20,
    "fail_reward": -20,
    "max_steps": 50
  }
)