import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

wine = pd.read_csv('https://bit.ly/wine-date')

#열에 대한 간단한 통계 산출 describe() 메서드
wine.describe()

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

#train_test_split의 기본 test set는 25%인데 지금 데이터가 충분히 많으므로 20%로 test size 지정
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

#전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

'''
#로지스틱 회귀 모델 훈련
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
'''

dt = DecisionTreeClassifier(random_state=42, max_depth =3)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(10,7))
#plot_tree(dt)   #dicision tree 전체 출력
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])      #max_depth=1 : decision tree 첫 번째만 출력
plt.show()

