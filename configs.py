import tensorflow as tf

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

STATE_SIZE = 192
ALLOWED_MOVES = ["L", "R", "U", "D", "F", "B",
                 "l", "r", "u", "d", "f", "b"]

model_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "hidden_layers": [STATE_SIZE, 100],
    "dtype": tf.float32,
    "optimizer": tf.train.AdamOptimizer,
    "learning_rate": 1e-3,
    "batch_size": 200
  }
)

agent_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "memory_size": 30000,
    "max_exploration_rate": 0.9,
    "min_exploration_rate": 0.05,
    "gamma": 0.9
  }
)

environment_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "allowed_moves": ALLOWED_MOVES,
    "step_reward": -0.1,
    "success_reward": 1,
    "fail_reward": -1,
    "max_steps": 50,
  }
)
