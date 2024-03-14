import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#도미 35마리의 길이와 무게 
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

#빙어 14마리의 길이와 무게
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.scatter(30,600)
plt.show()  #도미와 빙어 산점도 그래프에 찍어보기

'''
KNN 알고리즘을 이용해 도미와 빙어 데이터 구분
'''

#두 데이터를 하나의 리스트로 만들기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

#사이킷런을 이용하기 위해 각 특성의 리스트를 세로 방향으로 늘어뜨린 2차원 리스트로 변환
fish_data = [[l,w] for l, w in zip(length, weight)]
#첫 번째 데이터는 도미이고 두 번쨰 생선도 도미이다 라는 정답 데이터
fish_target = [1]*35 + [0]*14   #도미(1) 35개, 빙어(0) 14개

#KNeighborsClassifier의 객체 생성
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
plt.scatter(30,600)
plt.show()
print(kn.predict([[30,600]]))

# kn49 = KNeighborsClassifier(n_neighbors=49) #참고 데이터를 49개로 한 kn49 모델
# kn49.fit(fish_data, fish_target)
# kn49.score(fish_data,fish_target)
# print("kn49정확도 :",kn.score(fish_data, fish_target))
