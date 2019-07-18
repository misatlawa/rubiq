from attrdict import AttrDict

import tensorflow as tf


STATE_SIZE = 192
ALLOWED_MOVES = ['L', 'R', 'U', 'D', 'F', 'B',
                 'l', 'r', 'u', 'd', 'f', 'b']

model_config = AttrDict(
  {
    'state_size': STATE_SIZE,
    'action_size': len(ALLOWED_MOVES),
    'hidden_layers_size': [STATE_SIZE, STATE_SIZE, STATE_SIZE, 200],
    'dtype': tf.float32,
    'optimizer': tf.train.RMSPropOptimizer,
    'learning_rate': 1e-4,
    'update_interval': 500,
    'gamma': 0.8,
    'logdir': 'logdir/DoubleDQN-RMS'
  }
)

agent_config = AttrDict(
  {
    'state_size': STATE_SIZE,
    'action_size': len(ALLOWED_MOVES),
    'memory_size': 300000,
    'batch_size': 1000,
    'max_exploration_rate': 1,
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
