from attrdict import AttrDict

import tensorflow as tf


STATE_SIZE = 192
ALLOWED_MOVES = ['L', 'R', 'U', 'D', 'F', 'B',
                 'l', 'r', 'u', 'd', 'f', 'b']

model_config = AttrDict(
  {
    'state_size': STATE_SIZE,
    'action_size': len(ALLOWED_MOVES),
    'hidden_layers_size': [3*STATE_SIZE, 3*STATE_SIZE, 3*STATE_SIZE, STATE_SIZE],
    'dtype': tf.float64,
    'optimizer': tf.train.RMSPropOptimizer,
    'learning_rate': 1e-5,
    'update_interval': 400,
    'gamma': 0.8,
    'logdir': 'logdir/DoubleDQN-RMS-ReLU-offline-400/'
  }
)

agent_config = AttrDict(
  {
    'state_size': STATE_SIZE,
    'action_size': len(ALLOWED_MOVES),
    'memory_size': 300000,
    'batch_size': 1000,
    'max_exploration_rate': 1.,
    'min_exploration_rate': 0.05,
  }
)

environment_config = AttrDict(
  {
    'state_size': STATE_SIZE,
    'allowed_moves': ALLOWED_MOVES,
    'step_reward': 0.,
    'success_reward': 1.,
    'fail_reward': -1.,
    'max_steps': 20,
  }
)
