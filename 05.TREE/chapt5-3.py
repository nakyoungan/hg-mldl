'''트리의앙상블'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

rf.fit(train_input, train_target)
#print(rf.feature_importances_)

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)

scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print("결정트리개수 500개, 학습률 0.2")
print("train:",np.mean(scores['train_score']), "test:",np.mean(scores['test_score']))