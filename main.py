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
#train['Пол']=train['Пол'].fillna('М')
#train=train.drop(columns=['ID_y'])

train['ID_y']=train['ID']

categorical_columns=['Пол','Семья',"Этнос","Национальность","Религия","Образование","Профессия","smoking_status","Алкоголь"]

train.columns = train.columns.str.replace('Частота пасс кур', 'smoking_intensity')
test.columns = test.columns.str.replace('Частота пасс кур', 'smoking_intensity')

train.columns = train.columns.str.replace('Статус Курения', 'smoking_status')
test.columns = test.columns.str.replace('Статус Курения', 'smoking_status')


'''train.Профессия.replace({"вооруженные силы": 'служащие',
                           "техники и младшие специалисты": 'дипломированные специалисты',
                           "ведение домашнего хозяйства": 'квалифицированные работники сельского хозяйства и рыболовного',
                           "операторы и монтажники установок и машинного оборудования": 'низкоквалифицированные работники',
                         "ремесленники и представители других отраслей промышленности": "дипломированные специалисты",
                           }, inplace=True)
test.Профессия.replace({"вооруженные силы": 'служащие',
                           "техники и младшие специалисты": 'дипломированные специалисты',
                           "ведение домашнего хозяйства": 'квалифицированные работники сельского хозяйства и рыболовного',
                           "операторы и монтажники установок и машинного оборудования": 'низкоквалифицированные работники',
                         "ремесленники и представители других отраслей промышленности": "дипломированные специалисты",
                           }, inplace=True)'''

train.Семья.replace({"вдовец / вдова": 'в разводе',
                           "гражданский брак / проживание с партнером": 'other',
                           "никогда не был(а) в браке": 'other',
                           "раздельное проживание (официально не разведены)": 'other',
                           }, inplace=True)
test.Семья.replace({"вдовец / вдова": 'в разводе',
                           "гражданский брак / проживание с партнером": 'other',
                           "никогда не был(а) в браке": 'other',
                           "раздельное проживание (официально не разведены)": 'other',
                           }, inplace=True)

train.smoking_status.replace({"Никогда не курил": 'Никогда не курил(а)',
                           }, inplace=True)
test.smoking_status.replace({"Никогда не курил": 'Никогда не курил(а)',
                           }, inplace=True)
'''train.Религия.replace({"Нет": 'Атеист / агностик',
                           "Ислам": 'Христианство',
                      'Индуизм':'Христианство'
                           }, inplace=True)
test.Религия.replace({"Нет": 'Атеист / агностик',
                           "Ислам": 'Христианство',
                      'Индуизм':'Христианство'
                           }, inplace=True)'''

train.Этнос.replace({"другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)": 'other',
                           "прочее (любая иная этно-расовая группа, не представленная выше)": 'other',
                           }, inplace=True)
test.Этнос.replace({"другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)": 'other',
                           "прочее (любая иная этно-расовая группа, не представленная выше)": 'other',
                           }, inplace=True)
#print(train['Этнос'].value_counts())

for i in range(len(train)):
    if train['Национальность'][i]!='Русские':
        train['Национальность'][i]='other'
for i in range(len(test)):
    if test['Национальность'][i]!='Русские':
        test['Национальность'][i]='other'



train=train.drop(columns=['Этнос'])
test=test.drop(columns=['Этнос'])
categorical_columns.remove('Этнос') #PROBABLY SAME AS FOR NEXT 2

train=train.drop(columns=['Национальность'])
test=test.drop(columns=['Национальность'])
categorical_columns.remove('Национальность') #USING IT LOWERS VALIDATION BY ~0.01

#train=train.drop(columns=['Религия'])
#test=test.drop(columns=['Религия'])
#categorical_columns.remove('Религия') #USING IT LOWERS VALIDATION BY ~0.01

#train=train.drop(columns=['Прекращение работы по болезни'])
#test=test.drop(columns=['Прекращение работы по болезни'])

#train=train.drop(columns=['Хроническое заболевание легких'])
#test=test.drop(columns=['Хроническое заболевание легких'])

#train=train.drop(columns=['Бронжиальная астма'])
#test=test.drop(columns=['Бронжиальная астма'])
#print(train.columns)
#train=train.drop(columns=['Туберкулез легких '])
#test=test.drop(columns=['Туберкулез легких '])

#train=train.drop(columns=['Травмы за год'])
#test=test.drop(columns=['Травмы за год'])

train=train.drop(columns=['ВИЧ/СПИД'])
test=test.drop(columns=['ВИЧ/СПИД'])

#train=train.drop(columns=['Спорт, клубы'])
#test=test.drop(columns=['Спорт, клубы'])

#train=train.drop(columns=['Религия, клубы'])
#test=test.drop(columns=['Религия, клубы'])

'''train.Религия.replace({"Нет": 'Атеист / агностик',
                           "Ислам": 'Христианство',
                      'Индуизм':'Христианство'
                           }, inplace=True)
test.Религия.replace({"Нет": 'Атеист / агностик',
                           "Ислам": 'Христианство',
                      'Индуизм':'Христианство'
                           }, inplace=True)
'''

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

'''for i in OHE:
    one_hot = pd.get_dummies(train[i])
    one_hot2=pd.get_dummies(test[i])
    #print(train)
    train = train.drop(i, axis=1)
    test = test.drop(i, axis=1)
    # Join the encoded df
    train = train.join(one_hot)
    test = test.join(one_hot2)
    #print(train)
    #break
#TODO ^ OHE
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
#print(test.isna())
sns.set_theme(font_scale=0.6)
plt.figure(figsize=(50, 50))
sns.heatmap(test.corr(), annot=True)
plt.tight_layout()
#plt.show()
#print(test.isna().sum())
sub=pd.read_csv('sample_solution.csv')

from catboost import CatBoostClassifier
X=train.drop(columns=to_drop)


Y=train.drop(columns=X.columns)
#print(Y.columns)
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
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
    models=['model0_0.9325842696629213.pkl','model1_1.0.pkl','model2_0.9629629629629629.pkl','model3_0.9.pkl','model4_0.8823529411764706.pkl']
    #0- unknown 1-default 2-max_depth=1 3-max_depth=1 4- max_depth=1
    sum_=0
    for i in range(5):
        with open(models[i],'rb') as f:
            model=pickle.load(f)
        pred=model.predict(test)
        sub[to_drop[i]]=pred
        #print(models[i][7:-4])
        sum_+=float(models[i][7:-4])
    print(sum_/5)
else:
    its=[1,6,2,4,37]
    for i in range(5):
        if val:
            y_train.columns = y_train.columns.str.replace('ID_y', 'ID')
            X_temp=X_train.merge(y_train,on='ID')

            X1=X_temp[X_temp[to_drop[i]]==1]
            X2=X_temp[X_temp[to_drop[i]]==0]
            fraction=len(X2)//len(X1)
            val_tests=np.zeros(len(X_test))
            pred=np.zeros(len(test))
            print(len(X1),len(X2),'-@@-')
            for z in range(fraction):
                X2_2=X2[z*len(X1):(z+1)*len(X1)]
                X_fin=pd.concat([X1,X2_2])
                print(z*len(X1),(z+1)*len(X1))
                X_tr=X_fin.drop(columns=y_train.columns)
                y_tr=X_fin.drop(columns=X_train.columns)

                model = CatBoostClassifier(iterations=300, use_best_model=True, eval_metric='Recall', random_state=seed,learning_rate=0.001)

                model.fit(X_tr,y_tr[to_drop[i]],eval_set=(X_test,y_test[to_drop[i]]),verbose=0)
                val_test=np.array(model.predict(X_test))
                val_tests=val_test+val_tests

                test_pr=model.predict(test)
                pred=pred+test_pr
                best_its.append(
                    (model.best_score_['learn']['Recall'], model.best_score_['validation']['Recall'],
                     model.best_iteration_))
                with open(f'model{i}_{model.best_score_["validation"]["Recall"]}.pkl', 'wb') as f:
                    pickle.dump(model, f)
            val_tests=val_tests/fraction
            val_tests=list(map(lambda x: round(x),val_tests))
            print()
            print(recall_score(y_test[to_drop[i]],val_tests),'-------recall',i)
            pred=pred/fraction
            pred = list(map(lambda x: round(x), pred))
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
        #pred=model.predict(test)
        sub[to_drop[i]]=pred

    if val:
        sum_=0
        print(best_its)
        for g in best_its:
            sum_+=g[1]
        print(sum_/len(best_its))
sub.to_csv('final.csv',index=False)

#[(0.9658119658119658, 0.8736842105263158, 2), (1.0, 1.0, 12), (1.0, 0.8888888888888888, 12), (1.0, 0.75, 2), (1.0, 0.8235294117647058, 1)]
#[(0.8746438746438746, 0.8736842105263158, 0), (1.0, 1.0, 6), (0.9555555555555556, 0.9259259259259259, 2), (0.9210526315789473, 0.85, 4), (1.0, 0.8235294117647058, 37)]
#TODO mayvbe 4th column be predicted by seed 42
