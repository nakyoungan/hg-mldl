import numpy as np  
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


fruits = np.load('fruits_300.py')
fruits_2d = fruits.reshape(-1,100*100)

# imshow로 numpy 배열로 저장된 이미지 출력(밝을수록 255 짙을수록 0)
plt.imshow(fruits[0], cmap='gray')
plt.show()

# cmap = 'gray_r'로 흑백을 반전(밝을수록 0 짙을수록 255)
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

# 파인애플, 바나나 이미지 출력
fig,axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

apple = fruits[:100].reshape(-1,100*100)
pineapple = fruits[100:200].reshape(-1,100*100)
banana = fruits[200:300].reshape(-1,100*100)

apple_mean = np.mean(apple, axis=0).reshape(100,100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100,100)
banana_mean = np.mean(banana, axis=0).reshape(100,100)

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')
        axs[i,j].axis('off') # 축을 안보이게 함
plt.show()