from attrdict import AttrDict

import tensorflow as tf


STATE_SIZE = 192
ALLOWED_MOVES = ["L", "R", "U", "D", "F", "B",
                 "l", "r", "u", "d", "f", "b"]

model_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "hidden_layers": [STATE_SIZE, STATE_SIZE, STATE_SIZE, 200],
    "dtype": tf.float32,
    "optimizer": tf.train.AdamOptimizer,
    "learning_rate": 1e-4,
    "clip_norm": 1.,
    "batch_size": 1000,
    "weight_update_interval": 300,
    "logdir": "logdir/DoubleDQN-tanh"
  }
)

agent_config = AttrDict(
  {
    "state_size": STATE_SIZE,
    "action_size": len(ALLOWED_MOVES),
    "memory_size": 300000,
    "max_exploration_rate": 0.1,
    #"min_exploration_rate": 0.05,
    "gamma": 0.9,
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
