import tensorflow as tf

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

STATE_SIZE = 6*6*9
ALLOWED_MOVES = ["L", "R", "U", "D", "F", "B",
                 "L'", "R'", "U'", "D'", "F'", "B'"]

model_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "hidden_layers": [100, 100, 100],
    "dtype": tf.float32,
    "optimizer": tf.train.GradientDescentOptimizer,
    "learning_rate": 1e-3,
  }
)

agent_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "memory_size": 10000,
    "exploration_rate": 1,
    "gamma": 0.9
  }
)

environment_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "allowed_moves": ALLOWED_MOVES,
    "step_reward": -1,
    "success_reward": 20,
    "fail_reward": -20,
    "max_steps": 50,
  }
)