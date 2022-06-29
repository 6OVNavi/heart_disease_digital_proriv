import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random,os
seed=0
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed)

train=pd.read_csv('train.csv')
test=pd.read_csv('test_dataset_test.csv')

#train=train.drop(columns=['ID_y'])

train['ID_y']=train['ID']

categorical_columns=['Пол','Семья',"Этнос","Национальность","Религия","Образование","Профессия","Статус Курения","Алкоголь"]

train.columns = train.columns.str.replace('Частота пасс кур', 'smoking_intensity')
test.columns = test.columns.str.replace('Частота пасс кур', 'smoking_intensity')

train.smoking_intensity.replace({"1-2 раза в неделю": 1.5,
                           "3-6 раз в неделю": 4.5,
                           "не менее 1 раза в день": 7,
                           "4 и более раз в день": 28,
                           "2-3 раза в день":17.5 ,
                           }, inplace=True)
test.smoking_intensity.replace({"1-2 раза в неделю": 1.5,
                           "3-6 раз в неделю": 4.5,
                           "не менее 1 раза в день": 7,
                           "4 и более раз в день": 28,
                           "2-3 раза в день":17.5 ,
                           }, inplace=True)

train['smoking_intensity']=train['smoking_intensity'].fillna(1.5)
test['smoking_intensity']=test['smoking_intensity'].fillna(1.5)

train=train.drop(columns=['smoking_intensity'])
test=test.drop(columns=['smoking_intensity'])

from sklearn.preprocessing import LabelEncoder

'''for i in range(len(train)):
    print(train["Статус Курения"][i],train['Возраст курения'][i])
'''
#print(train['Возраст курения'].max())
train['Возраст курения']=train['Возраст курения'].fillna(58)
test['Возраст курения']=test['Возраст курения'].fillna(58)

#train=train.drop(columns=['Возраст курения'])
#test=test.drop(columns=['Возраст курения'])

train['Сигарет в день']=train['Сигарет в день'].fillna(0)
test['Сигарет в день']=test['Сигарет в день'].fillna(0)
#print(train['Возраст алког'].max())
train['Возраст алког' ]=train['Возраст алког'].fillna(63)
test['Возраст алког']=test['Возраст алког'].fillna(63)

#train=train.drop(columns=['Возраст алког'])
#test=test.drop(columns=['Возраст алког'])

#print(train['Время засыпания'])
train['Время засыпания'] = train['Время засыпания'].apply(lambda x: int(x.split(':')[0])+int(x.split(':')[1])/60).astype(float)
test['Время засыпания'] = test['Время засыпания'].apply(lambda x: int(x.split(':')[0])+int(x.split(':')[1])/60).astype(float)
#print(train['Время засыпания'])
train['Время пробуждения'] = train['Время пробуждения'].apply(lambda x: int(x.split(':')[0])+int(x.split(':')[1])/60).astype(float)
test['Время пробуждения'] = test['Время пробуждения'].apply(lambda x: int(x.split(':')[0])+int(x.split(':')[1])/60).astype(float)

#train['ID_1'] = train['ID'].apply(lambda x: int(x.split('-')[1])) good for last target
#test['ID_1'] = test['ID'].apply(lambda x: int(x.split('-')[1]))
#categorical_columns.append('ID_1')
#train['not_underage']=train['not_underage']

for col in train.columns:
    print(f'"{col}"',':',train[col].unique())

'''for i in range(5):
    col=train.columns[-i]
    #for j in range()
'''


for i in range(len(categorical_columns)):
    enc = LabelEncoder()
    if train[categorical_columns[i]].isna().sum()>0:
        train[categorical_columns[i]] = train[categorical_columns[i]].fillna(
            train[categorical_columns[i]].value_counts().idxmax())
    if test[categorical_columns[i]].isna().sum()>0:
        test[categorical_columns[i]] = test[categorical_columns[i]].fillna(
            test[categorical_columns[i]].value_counts().idxmax())
    enc.fit(np.unique(train[categorical_columns[i]].astype(str).unique().tolist()
                      + test[categorical_columns[i]].astype(str).unique().tolist()))
    train[categorical_columns[i]] = enc.transform(train[categorical_columns[i]].astype(str))
    test[categorical_columns[i]] = enc.transform(test[categorical_columns[i]].astype(str))
# TODO ^ FILLING CATEGORICALS
to_drop=["Артериальная гипертензия","ОНМК","Стенокардия, ИБС, инфаркт миокарда","Сердечная недостаточность","Прочие заболевания сердца",'ID_y']
temp=train[to_drop]
train=train.drop(columns=to_drop)
train[to_drop]=temp

sns.set_theme(font_scale=0.6)
plt.figure(figsize=(50, 50))
sns.heatmap(train.corr(), annot=True)
plt.tight_layout()
#plt.show()

each_cols=[]
corr_matrix=train.corr()
for i in range(5):
    arr=[]
    for j in range(len(train.columns)-5):
        if abs(corr_matrix.iloc[-i][j])>=0.01:
            arr.append(train.columns[j])
    each_cols.append(arr)
#print(each_cols)

#plt.savefig('corrplot2.png')
sub=pd.read_csv('sample_solution.csv')

from catboost import CatBoostClassifier
X=train.drop(columns=to_drop)
Y=train.drop(columns=X.columns)
#print(Y.columns)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
best_its=[]
X_test = X_test.drop(columns=['ID'])
y_test = y_test.drop(columns=['ID_y'])
cols=X_train.columns
import pickle
val=True
load_mode=False
test=test.drop(columns=['ID'])
if load_mode:
    models=['model0_0.8736842105263158.pkl','model1_1.0.pkl','model2_0.9629629629629629.pkl','model3_0.85.pkl','model4_0.8235294117647058.pkl']
    for i in range(5):
        with open(models[i],'rb') as f:
            model=pickle.load(f)
        pred=model.predict(test)
        sub[y_train.columns[i+1]]=pred
else:
    its=[1,6,2,4,37]
    for i in range(5):
        if val:
            model=CatBoostClassifier(iterations=300,use_best_model=True,eval_metric='Recall',random_state=seed,learning_rate=0.001)
            y_train.columns = y_train.columns.str.replace('ID_y', 'ID')
            X_temp=X_train.merge(y_train,on='ID')

            X1=X_temp[X_temp[to_drop[i]]==1]
            X2=X_temp[X_temp[to_drop[i]]==0]
            X2=X2.sample(n=int(len(X1)))

            X_fin=pd.concat([X1,X2])

            X_tr=X_fin.drop(columns=y_train.columns)
            y_tr=X_fin.drop(columns=X_train.columns)

            model.fit(X_tr,y_tr[to_drop[i]],eval_set=(X_test,y_test[to_drop[i]]),verbose=0)

            best_its.append(
                (model.best_score_['learn']['Recall'], model.best_score_['validation']['Recall'], model.best_iteration_))

            with open(f'model{i}_{model.best_score_["validation"]["Recall"]}.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            model = CatBoostClassifier(iterations=its[i], random_state=seed)

            # print(y_train.columns[i])
            X1=X_temp[X_temp[to_drop[i]]==1]
            X2=X_temp[X_temp[to_drop[i]]==0]
            X2 = X2.sample(n=len(X1))
            # print(X1,X2)
            # print(X1.columns,X2.columns)
            X_fin = pd.concat([X1, X2])

            #X_fin=X_fin.drop(columns=['ID'])
            X_tr = X_fin.drop(columns=y_train.columns)
            y_tr = X_fin.drop(columns=X_train.columns)
            X_tr=X_tr.drop(columns=['ID'])
            #print(X_tr,y_tr)
            model.fit(X_tr, y_tr[to_drop[i]], verbose=1)
        pred=model.predict(test)
        sub[to_drop[i]]=pred

    if val:
        print(best_its)
        print((best_its[0][1]+best_its[1][1]+best_its[2][1]+best_its[3][1]+best_its[4][1])/5)
sub.to_csv('final.csv',index=False)

#[(0.9658119658119658, 0.8736842105263158, 2), (1.0, 1.0, 12), (1.0, 0.8888888888888888, 12), (1.0, 0.75, 2), (1.0, 0.8235294117647058, 1)]
#[(0.8746438746438746, 0.8736842105263158, 0), (1.0, 1.0, 6), (0.9555555555555556, 0.9259259259259259, 2), (0.9210526315789473, 0.85, 4), (1.0, 0.8235294117647058, 37)]
#TODO mayvbe 4th column be predicted by seed 42
