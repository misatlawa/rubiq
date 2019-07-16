from attrdict import AttrDict

import tensorflow as tf


STATE_SIZE = 192
ALLOWED_MOVES = ["L", "R", "U", "D", "F", "B",
                 "l", "r", "u", "d", "f", "b"]

model_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "hidden_layers": [2*STATE_SIZE, 3*STATE_SIZE, 4*STATE_SIZE, 3*STATE_SIZE, 2*STATE_SIZE],
    "dtype": tf.float32,
    "optimizer": tf.train.AdamOptimizer,
    "learning_rate": 1e-4,
    "batch_size": 10**3,
    "update_interval": 500,
    "gamma": 0.9,
    "logdir": "logdir/DoubleDQN"
  }
)

agent_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "memory_size": 10**6,
    "max_exploration_rate": 1,
    "min_exploration_rate": 0.05,
    "max_steps": 50,
  }
)

environment_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "allowed_moves": ALLOWED_MOVES,
    "step_reward": -0.1,
    "success_reward": 1.,
  }
)
