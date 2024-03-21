import numpy as np  
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


fruits = np.load('fruits_300.py')
fruits_2d = fruits.reshape(-1,100*100)

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(np.unique(km.labels_, return_counts=True))

'''
클러스터별 이미지 생성 함수 생성

def draw_fruits(arr, ratio=1) :
    n = len(arr) # n은 샘플 개수
    # 한 행에 10개씩 이미지 생성
    rows = int(np.ceil(n/10))
    # 행이 1개면 열의 개수는 샘플 개수, 그렇지 않으면 10개
    cols = n if rows < 2 else 10
    
    fig,axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n :
                axs[i,j].imshow(arr[i*10+j], cmap='gray_r')
            axs[i,j].axis('off')
    plt.show()

#각 라벨별 클러스터 이미지 추출
draw_fruits(fruits[km.labels_==0])
draw_fruits(fruits[km.labels_==1])
draw_fruits(fruits[km.labels_==2])
'''

inertia= []
for k in range(2,7) :
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2,7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()