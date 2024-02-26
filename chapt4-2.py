import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier  #확률적 경사하강법을 제공하는 대표적인 분류용 클래스
import numpy as np
import matplotlib.pyplot as plt

fish = pd.read_csv('https://bit.ly/fish_csv')

#Species 열을 제외한 나머지 5개를 입력데이터로
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

ss=StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

'''
sc=SGDClassifier(loss='log_loss', max_iter=10, random_state=42)  #loss:손실함수(log:로지스틱), max_iter:에폭 횟수
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

#이어서 추가학습 1에폭 추가
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
'''


#적당한 에폭 찾기
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42) 
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled,train_target))
    test_score.append(sc.score(test_scaled,test_target))

plt.plot(train_score)
plt.plot(test_score)
plt.show()
#100번이 적당하군!


#tol : 일정 에포크 동안 성능이 향상되지 않으면 더 훈련하지 않고 자동으로 멈추는데 tol을 none으로 설정하여 무조건 100번 반복하도록 함.
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42) 
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))