from .movegen import movegen, get_zero_cube
import tensorflow as tf


class TensorCube:
  def __init__(self):
    self.session = tf.Session()
    zerocube = tf.constant(get_zero_cube(), name='zerocube')
    movements_dict = movegen()
    movement_list = [movements_dict[move] for move in 'RrLlFfBbUuDd']
    movement_tensors = [tf.constant(m, name='mov') for m in movement_list]
    move_vector = tf.placeholder(dtype='int64', shape=(None,), name='move_vector')
    move_tensor = tf.gather(movement_tensors, move_vector, name='move_tensor')
    state_tensor = tf.placeholder(dtype='int64', shape=(None, None, None), name='state_tensor')
    out_state_tensor = tf.matmul(move_tensor, state_tensor, name='out_state_tensor')
    tf.reduce_all(tf.equal(out_state_tensor, zerocube), axis=(1, 2), name='solved_vector')

  def action(self, moves, states):
    return self.session.run(
      fetches=(
        'solved_vector:0',
        'out_state_tensor:0'
      ),
      feed_dict={
        'move_vector:0': moves,
        'state_tensor:0': states
      }
    )