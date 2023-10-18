import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def preprocess(data):
    data.columns = data.columns.str.lower()
    data.drop(columns='unnamed: 0', inplace=True)
    data['ram']=data['ram'].str.replace('GB',"")
    data['ram'].astype('int32')
    data['weight']=data['weight'].str.replace('kg',"")
    data['weight'].astype('float32')
    data['touchscreen']=data['screenresolution'].apply(lambda x: 1 if "Touchscreen" in x else 0)
    data['ips']=data['screenresolution'].apply(lambda x: 1 if "IPS" in x else 0)
    res=data['screenresolution'].str.split('x',n=1, expand=True)
    data['X_res']=res[0]
    data['Y_res']=res[1]
    data['X_res']=data['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
    data['X_res']=data['X_res'].astype('int')
    data['Y_res']=data['Y_res'].astype('int')
    data['ppi']=(((data['X_res']**2 )+ (data['Y_res']**2))**0.5 / data['inches']).astype('float')
    data.drop(columns=['screenresolution','inches','X_res','Y_res'], inplace=True)
    data['processor']=data['cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    data['cpu_brand']=data['processor'].apply(processor_info)
    data.drop(columns=['processor','cpu'], inplace=True)

    data['ssd'] = data['memory'].str.extract('(\d+)GB SSD').fillna(0).astype(int)
    data['hdd'] = data['memory'].str.extract('(\d+)TB HDD').fillna(0).astype(int) * 1000
    data['flash_storage'] = data['memory'].str.extract('(\d+)GB Flash Storage').fillna(0).astype(int)
    data['hybrid'] = data['memory'].str.extract('(\d+\.?\d*)TB Hybrid').fillna(0).astype(float).astype(int) * 1000
    data.drop(columns=['memory','flash_storage','hybrid'], inplace=True)
    data['gpu_type']=data['gpu'].apply(lambda x:x.split()[0])
    data=data[data['gpu_type']!='ARM']
    data['os']=data['opsys'].apply(os_type)
    data.drop(columns=['opsys','gpu'], inplace=True)

    return data

def os_type(word):
    if word=='Windows 10' or word=='Windows 7' or word=='Windows 10':
        return 'Windows'
    elif word=='macOS' or word=='Mac OS X':
        return 'Mac'
    elif word=='Linux':
        return 'Linux'
    else:
        return 'Others/No OS'
    
def processor_info(word):
    if word=="Intel Core i5" or word=="Intel Core i7" or word=="Intel Core i3":
        return word
    else:
        if word.split()[0]=="Intel":
            return "Other Intel Processor"
        else:
            return "AMD processor"

def predict(data,features):
    X = data.drop(columns='price')
    y = np.log(data['price'])
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=2)
    random_forest_model = RandomForestRegressor()

    # Create the pipeline for the Random Forest model
    step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])],
                            remainder='passthrough')
    pipe = Pipeline([('step1', step1), ('random_forest', random_forest_model)])

    # Fit the pipeline and make predictions
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    query=features.reshape(1,12)
    y_pred1 = pipe.predict(query)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("r2", r2)
    print('MAE', mae)
    return int(np.exp(y_pred1)[0])




