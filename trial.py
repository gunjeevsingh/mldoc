##use disease.pkl
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np # linear algebra
import pandas as pd


vectorizer = CountVectorizer()
data1= pd.read_csv('dataset.csv')
data2= pd.read_csv('symptom_precaution.csv')
data3= pd.read_csv('Symptom-severity.csv')
data4= pd.read_csv('symptom_Description.csv')
data4.loc[16,'Disease'] = 'Dimorphic hemmorhoids(piles)'
data3.loc[102,'Symptom'] = '_patches'
X = data1.iloc[:,1:]
y = data1.iloc[:,0]

def combine(symptoms_list):
    symptoms_list = [x for x in list(symptoms_list) if  isinstance(x, str)]
    return ' '.join(symptoms_list)

X['symp'] = [combine(x) for x in  X.values]

XX = X['symp']

X = X['symp']

X = vectorizer.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
yy = le.fit_transform(y)





model=pickle.load(open('disease.pkl','rb'))

symptoms = ['patches anxiety belly_pain irritability palpitations slurred_speech abdominal_pain']
test = vectorizer.transform(symptoms).toarray()

y = model.predict(test)
z = model.predict_proba(test)

data2.set_index('Disease' , inplace=True)
data4.set_index('Disease' , inplace=True)
disease = le.classes_[y]

disease_info = data4.loc[disease[0]].values[0]
precaution = data2.loc[disease[0]].values[0]

n_data = pd.DataFrame()
n_data['Disease'] = data1['Disease']
n_data['symp'] = XX

    
results = list(z[0])
results.sort()
most_common = list(dict.fromkeys(results[::-1][:5]))
diseases_ = []
proba_disease = []
for i in most_common:
    index = [indx for indx , v in enumerate(list(z[0])) if v == i ]
    for indx in index:
        disease = le.classes_[indx]
        proba = round(i*100,3)
        diseases_.append(disease)
        proba_disease.append(proba)
    
    

def reults(diseases_ , proba_disease):
    dis_proba = list(zip(diseases_,proba_disease))
    for disease,proba in dis_proba[:4]:
        print(f'[+] PREDICTED DISEASE ({proba}%) : ' , disease)
        try:
            info = data4[data4.index == disease.strip()]['Description'].values[0]
        except:
            info = data4[data4.index == disease]['Description'].values[0]
        print('[!] ABOUT DISEASE : ' , info)

        precautions = ', '.join([x for x in data2.loc[disease].values if isinstance(x , str)])
        print('[X] PRECAUTIONS : ',precautions)

        print('\n')

def risk_calc(symptoms):
    serius_degrees = [data3[data3['Symptom'] == symp].values[0][1] for symp in symptoms if symp in data3['Symptom'].values]
    symp_degree = list(zip(symptoms , serius_degrees))
    rsik_score = sum([x for x in serius_degrees])
    most_serious = ', '.join([x[0] for x in symp_degree if x[1] >= 5])
    print('[+] Risk Score :>> ' , rsik_score)

    if rsik_score in range(15,25):
        print('[-] Plase consult doctor')
    elif rsik_score >= 25:
        print('[-] Plase Go to the nearest emergency department')
    elif rsik_score < 15:
        print("[-] Your risk score is low , so you don't need to take any action! Your symptoms are mild and will disappear in a few hours.")


    print(f'[!] Most Serious Symptoms : {most_serious.replace("_" , " ")}')
    print(f'[!] Risk Score for each Symptom : {symp_degree}')


reults(diseases_ , proba_disease)
risk_calc(symptoms)