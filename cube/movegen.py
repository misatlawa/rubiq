from functools import reduce
import numpy as np


def movegen():
  csvs = ['RO', 'GB', 'YW', 'OTurn']

  basic_movements = [
    np.genfromtxt('cube/matrices/{}.csv'.format(x), delimiter=',', dtype=int) for x in csvs
  ]

  basic_movements_inv = [np.linalg.inv(x).astype(int) for x in basic_movements]

  basic_movements += basic_movements_inv

  sequences = {
    'R': [3],
    'r': [7],
    'L': [2, 2, 3, 6, 6],
    'l': [2, 2, 7, 6, 6],
    'U': [1, 3, 5],
    'u': [1, 7, 5],
    'D': [5, 3, 1],
    'd': [5, 7, 1],
    'F': [6, 3, 2],
    'f': [6, 7, 2],
    'B': [2, 3, 6],
    'b': [2, 7, 6],
  }

  movements = {x: reduce(np.dot, [basic_movements[i] for i in y[::-1]]) for x, y in sequences.items()}

  return movements


def get_zero_cube():
  return np.genfromtxt('cube/matrices/ZEROCUBE.csv', delimiter=',', dtype=int)
