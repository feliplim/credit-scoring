from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from datetime import date, timedelta
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
import gzip
import io

def read(file_path):
    with gzip.open(file_path, 'rb') as file:
        content = file.read().decode('utf-8')
    data = pd.read_csv(io.StringIO(content))
    data = data.replace([np.inf, -np.inf], np.nan)
    return data

def impute(data):
    idx = data[['SK_ID_CURR']]
    features = data.drop(columns=['SK_ID_CURR'])
    features_names = data.drop(columns=['SK_ID_CURR']).columns.to_list()

    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

    imp_median.fit(features)

    features_fill = imp_median.transform(features)
    features_fill = pd.DataFrame(features_fill, columns=features_names)

    df = pd.concat([idx, features_fill], axis=1)

    return df

def scale(data):
    idx = data[['SK_ID_CURR']]
    features = data.drop(columns=['SK_ID_CURR'])
    features_names = data.drop(columns=['SK_ID_CURR']).columns.to_list()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(df_scaled, columns=features_names)

    df = pd.concat([idx, df_scaled], axis=1)

    return df

# Set FastAPI app
app = FastAPI(title='Home Credit Default Risk', 
              description='Get information related to the probability of a client not repaying a loan', 
              version='0.1.0')

# Set global variables
N_NEIGHBORS = 50
CUSTOM_THRESHOLD = 0.274

# Get dataframes
clients_to_predict = read('../data/processed/test_feature_engineering_encoded.csv.gz')
current_clients = read('../data/processed/train_feature_engineering_encoded_extract.csv.gz')

# Prepare dataframes
clients_to_predict_fill = impute(clients_to_predict)

# Scale 
clients_to_predict_scaled = scale(clients_to_predict_fill)

# Load model
lgbm = joblib.load('../models/lightgbm_classifier.pkl')

# Load shap model
lgbm_shap = joblib.load('../models/lightgbm_shap_explainer.pkl')
shap_values = lgbm_shap.shap_values(clients_to_predict.drop(columns=['SK_ID_CURR']))

# Classification model
knn = NearestNeighbors(n_neighbors=N_NEIGHBORS+1, algorithm='auto', n_jobs=-1, metric='cosine')
knn.fit(clients_to_predict_scaled.drop(columns=['SK_ID_CURR']))

@app.get('/')
async def read_root():
    return 'Home Credit Default Risk API'

@app.get('/api/clients')
async def get_clients_id():
    '''
    Endpoint to get all clients id
    '''
    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()
    return {'clientsID': clients_id}

@app.get('/api/clients/{id}/personal_information')
async def get_client_personal_information(id: int):
    '''
    Endpoint to get client's information
    '''
    PERSONAL_INFORMATION = ['SK_ID_CURR', 'CODE_GENDER', 'CNT_CHILDREN',  'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_BIRTH']
    FAMILY_STATUS = clients_to_predict.filter(regex='^NAME_FAMILY_STATUS_').columns.tolist()
    family_status = [col.replace('NAME_FAMILY_STATUS_', '') for col in FAMILY_STATUS]

    EDUCATION_TYPE = clients_to_predict.filter(regex='^NAME_EDUCATION_TYPE_').columns.tolist()
    education_type = [col.replace('NAME_EDUCATION_TYPE_', '') for col in EDUCATION_TYPE]

    df_client_info = clients_to_predict[PERSONAL_INFORMATION + FAMILY_STATUS + EDUCATION_TYPE][clients_to_predict['SK_ID_CURR'] == id]

    for col in df_client_info.columns:
        globals()[col] = df_client_info.iloc[0, df_client_info.columns.get_loc(col)]

    civilStatus = ''
    for col, status in zip(FAMILY_STATUS, family_status):
        if globals()[col] == 1: 
            civilStatus = status

    educationType = ''
    for col, education in zip(EDUCATION_TYPE, education_type):
        if globals()[col] == 1:
            educationType = education

    client_info = {
            'clientId' : int(SK_ID_CURR),
            'gender' : 'Man' if int(CODE_GENDER) == 0 else 'Woman',
            'countChildren' : 'No children' if int(CNT_CHILDREN) == 0 else int(CNT_CHILDREN),
            'ownCar' : 'No car' if int(FLAG_OWN_CAR) == 0 else 'Yes',
            'ownRealty' : 'No realty' if int(FLAG_OWN_REALTY) == 0 else 'Yes', 
            'age': round(float(DAYS_BIRTH)/-365), 
            'birthday': date.today() - timedelta(days=float(DAYS_BIRTH)/-1),
            'civilStatus': civilStatus, 
            'educationType': educationType
    }

    return client_info

@app.get('/api/clients/{id}/bank_information')
async def get_client_bank_information(id: int):
    '''
    Endpoint to get client's information
    '''
    BANK_INFORMATION = ['SK_ID_CURR', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 
                    'AMT_ANNUITY','DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'PAYMENT_RATE', 'CREDIT_INCOME_PERCENT']
    INCOME_TYPE = clients_to_predict.filter(regex='^NAME_INCOME_TYPE_').columns.tolist()
    income_type = [col.replace('NAME_INCOME_TYPE_', '') for col in INCOME_TYPE]

    df_client_info = clients_to_predict[BANK_INFORMATION + INCOME_TYPE][clients_to_predict['SK_ID_CURR'] == id].fillna(0)

    for col in df_client_info.columns:
        globals()[col] = df_client_info.iloc[0, df_client_info.columns.get_loc(col)]

    incomeType = ''
    for col, income in zip(INCOME_TYPE, income_type):
        if globals()[col] == 1:
            incomeType = income

    client_info = {
            'clientId' : int(SK_ID_CURR),
            'totalIncome': round(float(AMT_INCOME_TOTAL)), 
            'extSource1': float(EXT_SOURCE_1),
            'extSource2': float(EXT_SOURCE_2),
            'extSource3': float(EXT_SOURCE_3), 
            'seniority': round(float(DAYS_EMPLOYED)/-365), 
            'registrationSince': date.today() - timedelta(days=float(DAYS_REGISTRATION)/-1), 
            'amtCredit': round(float(AMT_CREDIT)), 
            'annualCredit': round(float(AMT_ANNUITY)), 
            'lengthCredit': round(12*(float(AMT_CREDIT) / float(AMT_ANNUITY))),
            'paymentRate': round(100*float(PAYMENT_RATE), 2), 
            'creditIncomeRatio': round(float(CREDIT_INCOME_PERCENT), 2),             
            'incomeType': incomeType
    }
    return client_info

@app.get('/api/clients/{id}/prediction')
async def get_prediction(id: int):
    '''
    EndPoint to get the probability honor/compliance of a client
    '''
    df_client_info = clients_to_predict[clients_to_predict['SK_ID_CURR'] == id]
    df_client_info = df_client_info.drop(df_client_info.columns[[0]], axis=1)

    result_proba = lgbm.predict_proba(df_client_info)
    y_prob = result_proba[:, 1]

    result = (y_prob >= CUSTOM_THRESHOLD).astype(int)

    if (int(result[0]) == 0):
        result = 'Yes'
    else:
        result = 'No'    

    client_info = {
        'clientId': id, 
        'repay' : result,
        'score' : round(1000*result_proba[0][0]),
        'probability0' : result_proba[0][0],
        'probability1' : result_proba[0][1],
        'threshold' : CUSTOM_THRESHOLD
    }
    return client_info

@app.get('/api/clients/{id}/prediction/shap/local')
async def get_local_shap(id: int):

    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        shap_values_idx = shap_values[0][idx, :]
        shap_values_abs_sum = np.abs(shap_values_idx)
        top_feature_indices = np.argsort(shap_values_abs_sum)[-10:]
        top_feature_names = clients_to_predict.drop(columns=['SK_ID_CURR']).columns[top_feature_indices]
        top_feature_shap_values = shap_values_idx[top_feature_indices]


    client_shap = {
        #'clientId': id, 
        #'shapPosition': idx, 
    }

    for name, value in zip(top_feature_names, top_feature_shap_values):
        client_shap[name] = value

    return client_shap

@app.get('/api/clients/{id}/prediction/shap/global')
async def get_global_shap(id: int):

    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        feature_names = clients_to_predict.drop(columns=['SK_ID_CURR']).columns
        shap_values_summary = pd.DataFrame(shap_values[0], columns=feature_names)
        top_global_features = shap_values_summary.abs().mean().nlargest(10)

    client_shap = {
        #'clientId': id, 
        #'shapPosition': idx, 
    }

    client_shap.update(top_global_features.to_dict())

    return client_shap

@app.get('/api/clients/{id}/prediction/neighbors')
async def get_neighbors(id: int):

    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        client_idx = clients_to_predict_scaled.drop(columns=['SK_ID_CURR']).iloc[idx].values.reshape(1, -1)

        distances, indices = knn.kneighbors(client_idx)
        indices = [int(item) for item in indices[0]]
        indices = indices[1:]
        distances = [float(item) for item in distances[0]]
        distances = distances[1:]
        scores = [1-item for item in distances]

        neighbors = clients_to_predict_scaled.iloc[indices]['SK_ID_CURR'].tolist()

        result = dict(zip(neighbors, scores))

    return result