import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.py')

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[101:200].reshape(-1, 100*100)
banana = fruits[201:300].reshape(-1, 100*100)

print(apple.shape)