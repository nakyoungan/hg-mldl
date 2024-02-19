import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#농어 길이로 무게 맞추기
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

'''
#농어데이터 산점도 찍어보기
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
'''

#길이가 특성이고 무게가 타깃
#훈련세트와 테스트세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

#사이킷런 사용을 위해 배열 2차원으로 변경
#크기에 -1 지정하면 나머지 원소 개수로 모두 채우라는 의미 (첫번째는 원소 두번쨰는 1)
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

knr = KNeighborsRegressor()
lr = LinearRegression()

#KNN train
knr.fit(train_input, train_target)

'''
#R² 점수
print("R² 는",knr.score(test_input, test_target))

#테스트세트에 대한 예측 
test_prediction = knr.predict(test_input)

#테스트세트에 대한 평균 절댓값 오차 계산  
mae = mean_absolute_error(test_target, test_prediction)
print("MAE는",mae)
'''
'''
#훈련모델로 훈련 세트의 R²점수 
print("R² 는",knr.score(train_input, train_target))
해보니 underfitting 발생 > 해결:모델을 더 복잡하게 만들어야함
KNN 알고리즘에서 모델을 복잡하게 만들려면? 
> 이웃의 개수 K를 줄이기 
> 사이킷런의 기본값인 5를 3으로 변경
'''

knr.n_neighbors = 3

knr.fit(train_input, train_target)
print("R² 는",knr.score(train_input, train_target))

