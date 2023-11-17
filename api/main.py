from fastapi import FastAPI, File, HTTPException
import pandas as pd
import os
import joblib
from datetime import date, timedelta
import pickle
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import shap
from datetime import date, timedelta

# Set FastAPI app
app = FastAPI(title='Home Credit Default Risk', 
              description='Get information related to the probability of a client not repaying a loan', 
              version='0.1.0')

# Set global variables
N_CUSTOMERS = 100
N_NEIGHBORS = 50
CUSTOM_THRESHOLD = 0.27

# Set local directory
project_path = 'C:\\Users\\lipe_\\Documents\\projets\\credit-scoring\\'
os.chdir(project_path)

# Get dataframes
current_clients = pd.read_csv('data\\processed\\train_feature_engineering_encoded.csv', nrows=N_CUSTOMERS)
clients_to_predict = pd.read_csv('data\\processed\\test_feature_engineering_encoded.csv', nrows=N_CUSTOMERS)

# Load model
lgbm = joblib.load('models\\lightgbm_classifier.pkl')

# Load shap model
lgbm_shap = joblib.load('models\\lightgbm_shap_explainer.pkl')
shap_values = lgbm_shap.shap_values(clients_to_predict.drop(columns=['SK_ID_CURR']))

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
            'clientId' : float(SK_ID_CURR),
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
        'clientId': id, 
        'shapPosition': idx, 
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
        'clientId': id, 
        'shapPosition': idx, 
    }

    client_shap.update(top_global_features.to_dict())

    return client_shap

@app.get('/api/statistics/genders')
async def get_stats_gender():
    
    count_men = current_clients[current_clients['CODE_GENDER'] == 0]['SK_ID_CURR'].count()
    count_women = current_clients[current_clients['CODE_GENDER'] == 1]['SK_ID_CURR'].count()

    gender = {
        'countMen': int(count_men), 
        'countWomen': int(count_women)
    }

    return gender

@app.get('/api/statistics/ages')
async def get_stats_ages():

    df_ages = current_clients[['SK_ID_CURR', 'DAYS_BIRTH']]
    df_ages['age'] = round(df_ages['DAYS_BIRTH']/-365)
    df_ages.drop(columns=['DAYS_BIRTH'])

    ages = df_ages.set_index('SK_ID_CURR')['age'].to_dict()

    return ages

@app.get('/api/statistics/loans')
async def get_stats_loan():

    count_0 = current_clients[current_clients['TARGET'] == 0]['SK_ID_CURR'].count()
    count_1 = current_clients[current_clients['TARGET'] == 1]['SK_ID_CURR'].count()

    loans = {
        'repaid': int(count_0), 
        'defaulted': int(count_1)
    }

    return loans