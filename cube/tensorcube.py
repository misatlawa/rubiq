from movegen import movegen, get_zero_cube
import tensorflow as tf


class TensorCube:
  def __init__(self):
    self.session = tf.Session()
    movements_dict = movegen()
    movement_list = [movements_dict[move] for move in 'RrLlFfBbUuDd']
    movement_tensors = [tf.constant(m, name='mov') for m in movement_list]
    move_vector = tf.placeholder(dtype='int64', shape=(None,), name='move_vector')
    move_tensor = tf.gather(movement_tensors, move_vector, name='costam')
    state_tensor = tf.placeholder(dtype='int64', shape=(None, None, None), name='state_tensor')
    tf.matmul(move_tensor, state_tensor, name='out_state_tensor')

  def action(self, moves, states):
    return self.session.run('out_state_tensor:0', feed_dict={
      'move_vector:0': moves,
      'state_tensor:0': states})

def mini_test():
  kuba = TensorCube()
  zero = get_zero_cube()
  a = [zero, zero, zero]
  a = kuba.action([0,0,0], a)
  print([(x==zero).all() for x in a])
  a = kuba.action([1,0,1], a)
  print([(x==zero).all() for x in a])
