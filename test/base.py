import numpy as np

zc = np.genfromtxt('cube/matrices/ZEROCUBE.csv', delimiter=',', dtype=int)

turns = ['RO', 'GB', 'YW', 'OTurn']

turns_m = [
  np.genfromtxt('cube/matrices/{}.csv'.format(x), delimiter=',', dtype=int) for x in turns
]

ro, gb, yw, oturn = turns_m
ro_inv, gb_inv, yw_inv, oturn_inv = [np.linalg.inv(x).astype(int) for x in turns_m]


def is_solved(cc):
  return (cc == zc).all()