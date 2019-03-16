import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

ctx = './data/'
train = pd.read_csv(ctx+'train.csv')
test = pd.read_csv(ctx+'test.csv')

train.columns

"""
Index(['PassengerId', : 승객번호
'Survived', : 생존여부 (1: 생존, 0: 사망)
'Pclass', : 승선권클래스 (1~3: 1~3등석)
'Name', : 승객 이름
'Sex', : 승객 성별
'AbSp', : 동반한 형제자매, 배우자 수
'Pge', : 승객 나이
'Siarch', : 동반한 부모, 자식 수
'Ticket', : 티켓의 고유넘버
'Fare', : 티켓의 요금
'Cabin', : 객실 번호
'Embarked' : 승선한 항구명 (C : 쉐르부룩, Q: 퀸스타운, S: 사우스햄턴)
],
      dtype='object')
"""



# 생존여부와 조건과의 상관관계를 파악한다

# 1. 생존률 파악
f, ax = plt.subplots(1,2, figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data = train, ax=ax[1])
ax[1].set_title('Survived')


"""
탑승객의 61.6% 사망, 38.4% 생존
"""
# 2. 성별에 따른 생존률
f, ax = plt.subplots(1,2, figsize=(18,8))
train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0],shadow=True)
train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[1],shadow=True)
ax[0].set_title('male')
ax[1].set_title('female')


"""
남성: 생존 18.9%, 사망 81.1%
여성: 생존 74.2%, 사망 25.8%
"""
# 3. 승선권클래스에 따른 생존률

df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']
pd.crosstab(df_1, df_2, margins=True)
# 그래프가 아닌 판다스의 자체 table 기능을 사용한 분석

"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""

# 4. 배를 탄 항구에 따른 생존률 Embarked

f, ax = plt.subplots(2,2, figsize=(20,15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('No of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('Male - Female for Embarked')
sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',  data=train, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')


"""
절반이상의 승객이 S항구에서 배를 탔으며, 여기에서 탑승한 승객의 70% 가량이 남성임.
남성의 사망률이 여성보다 훨씬 높았기에, 자연스럽게 S항구에서 탑승한 승객의 사망률이
높게 나왔음.
C항구에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이 높은 것으로 보여서
이 도시는 부유한 도시라는 것을 예측할 수 있음.
"""

# **************
# 결과 도출을 위한 전처리 (pre-processing)
# *************

"""
가장 강한 상관관계를 가지고 있는 성별, 객실 등급, 탑승 항구 세가지 정보를 가지고 모델을 
구성하려고 함.
"""
# 결측치 제거
# 비어있는 데이터를 제거하여 연산에서의 오류를 방지함

train.info

# [891 rows x 12 columns]

train.isnull().sum()
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177  # 177개의 결측치가 있음
나이에 따른 생존여부가 상관관계가 있을 듯하여 임의의 데이터로 채워 넣어서 해결
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
# 객실번호에 따른 생존여부가 상관관계가 있을 듯하여 데이터를 채우려 했으나,
# 임의의 데이터를 산정하기 어렵고, Pclass 를 대체하여 분석가능하므로 제거함.
Embarked         2
# 승선한 항구의 결측치는 예상하기 어려우나 , 가장 많이 승선한 S항구로 임의값으로 대처함.
dtype: int64
"""



def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived',' Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))

bar_chart('Sex')
bar_chart('Pclass') # 생존자 1등석, 사망자 3등석
bar_chart('SibSp') # 동반한 형제자매, 배우자
bar_chart('Parch') # 동반한 부모, 자식수
bar_chart('Embarked')

# Cabin은  null 값이 많아서, Ticket 은 전혀 관계가 없어서 삭제

train = train.drop(['Cabin'], axis =1)
test = test.drop(['Cabin'], axis =1)
train = train.drop(['Ticket'], axis =1)
test = test.drop(['Ticket'], axis =1)
train.columns
test.columns
# Embarked 에 있는 null  2개 처리

s_city = train[train['Embarked']=='S'].shape[0]
print("S : ", s_city) # S :  644
c_city = train[train['Embarked']=='C'].shape[0]
print("C : ", c_city) # C :  168
q_city = train[train['Embarked']=='Q'].shape[0]
print("Q : ", q_city) # Q :  77
# na 에 값을 채우기
train = train.fillna({"Embarked":"S"})# {'' : ''}

# 머신러닝에서 모든 값은 숫자로만 인식함
# 따라서 S, C, Q 를 숫자로 1, 2, 3 으로 가공함

city_mapping = {"S":1, "C":2,"Q":3}
train["Embarked"] = train["Embarked"].map(city_mapping)
test["Embarked"] = test["Embarked"].map(city_mapping)


# Name 값 가공하기
"""
승객이름에서 Mr, Mrs 가 있습니다.
"""
combine = [train,test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])
"""
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
"""
# Mr, Mrs, Miss, Royal, Rare, Master 6개로 줄여서 정리

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')


train[['Title','Survived']].groupby(['Title'], as_index = False).mean()
# groupby 는 같은 값을 가지는 인스턴스를 묶어서 연산하라(mean 이면 평균구하기)
"""
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.250000
5   Royal  1.000000
"""
# 위 데이터를 바탕으로 1부터 6까지로 매핑을 숫자로 함.
title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # 결측치가 있으면 0


# Name 과 PassengerId 삭제
train = train.drop(['Name','PassengerId'], axis =1) # 한꺼번에 두개 삭제
#SVC() 에서 사용할 테스트 아이디 작성.... 검증 이후에 제출할 테스트값 으로 사용함
test_id = test['PassengerId']
# ----------테스트 ID ------------

test = test.drop(['Name','PassengerId'], axis =1)
train.columns
test.columns
combine = [train,test]

# 성별도 숫자로 치환
sex_mapping = {'male':0,'female':1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# 나이 age 가공
# age 에서 null 값에 대한 처리
# 일단 -0.5 로 채워 넣은 후 pandas 의 cut() 함수를 사용해서
# AgeGroup 를 만듦. cut() 함수는 각 구간의 값을 특정값으로
# 정의해주는 함수

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ["Unknown","Baby","Child","Teenager","Student","Young Adult","Adult","Senior"]
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)


bar_chart('AgeGroup')
# {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Rare':6}
age_title_mapping = {1:'Young Adult',
                     2:'Young Adult',3:'Adult',4:'Adult',5:'Adult',6:'Adult'}

for i in range(len(train['AgeGroup'])):
    if train['AgeGroup'][i] == "Unknown":
        train['AgeGroup'][i] = age_title_mapping[train['Title'][i]]
for i in range(len(test['AgeGroup'])):
    if test['AgeGroup'][i] == "Unknown":
        test['AgeGroup'][i] = age_title_mapping[test['Title'][i]]



# AgeGroup 를 숫자로 치환

age_mapping = {"Baby":1,"Child":2,"Teenager":3,"Student":4,"Young Adult":5,"Adult":6,"Senior":7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train = train.drop(['Age'],axis = 1)
test = test.drop(['Age'],axis = 1)



# 'Fare', : 티켓의 요금
# qcut 함수를 사용. 4개의 범위로 cut

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = {1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = {1,2,3,4})

train = train.drop(['Fare'],axis = 1)
test = test.drop(['Fare'],axis = 1)



# **************
# 데이터 모델링
# **************
train_data = train.drop('Survived', axis = 1)
target = train['Survived']

#train_data.shape, target.shape

# ((891, 8), (891,))
#print('----- 모델 정보 -------')
print(train.info)

# *************
# 모델 학습과 검증
# *************

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# KNN 방식 검증
print('>>>> KNN 방식 검증') #81.48
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score) * 100, 2))


# 결정트리 방식 검증
print('>>>> 결정트리 방식 검증') #79.13
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score) * 100, 2))

# 랜덤포레스트 방식 검증
print('>>>> 랜덤포레스트 방식 검증') #81.03
clf = RandomForestClassifier(n_estimators=13) #13개의 결정트리를 사용함
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score) * 100, 2))

# 나이브베이즈 방식 검증
print('>>>> 나이브베이즈 방식 검증') #79.57
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score) * 100, 2))

# SVM 방식 검증
print('>>>> SVM 방식 검증') #83.05
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score) * 100, 2))


# ******
# SVM 방식이 가장 정확도(accuracy) 가 높으므로 이 방식으로 파일을 제작하여 제출하기로 함
# ******

clf = SVC()
clf.fit(train_data, target)
prediction = clf.predict(test)
submission = pd.DataFrame(
    {'PassengerId' : test_id,
     'Survived' : prediction,
     }
)
print('=== 최종 제출할 모델===')
print(submission)
submission.to_csv(ctx + 'submission.csv', index=False)



