from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# print(train_input.shape, train_target.shape)
# print(test_input.shape, test_target.shape)

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')


print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))