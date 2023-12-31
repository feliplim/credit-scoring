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
N_NEIGHBORS = 1000
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

# Endpoints to get information about the current client
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
    ''' Endpoint to get local shap values
    '''
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


    client_shap = {}

    for name, value in zip(top_feature_names, top_feature_shap_values):
        client_shap[name] = value

    return client_shap

@app.get('/api/clients/{id}/prediction/shap/global')
async def get_global_shap(id: int):
    ''' Endpoint to get global shap values
    '''
    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        feature_names = clients_to_predict.drop(columns=['SK_ID_CURR']).columns
        shap_values_summary = pd.DataFrame(shap_values[0], columns=feature_names)
        top_global_features = shap_values_summary.abs().mean().nlargest(10)

    client_shap = {}

    client_shap.update(top_global_features.to_dict())

    return client_shap

# Endpoints to get information about the neighbors / similar clients
@app.get('/api/clients/{id}/prediction/neighbors')
async def get_neighbors(id: int):
    ''' Endpoint to get all neighbors of the current client and their similarity score
    '''
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

@app.get('/api/clients/{id}/prediction/neighbors/totalIncome')
async def get_neighbors_total_income(id: int):
    ''' Endpoint to get the total income of the neighbors of the current client
    '''
    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        client_idx = clients_to_predict_scaled.drop(columns=['SK_ID_CURR']).iloc[idx].values.reshape(1, -1)

        distances, indices = knn.kneighbors(client_idx)
        indices = [int(item) for item in indices[0]]
        indices = indices[1:]
        neighbors = clients_to_predict_scaled.iloc[indices]['SK_ID_CURR'].astype(int).tolist()
        client_info = []

        for neighbor in neighbors:
            globals()['df_' + str(neighbor)] = clients_to_predict[clients_to_predict['SK_ID_CURR'] == neighbor]
            globals()['df_' + str(neighbor)] = globals()['df_' + str(neighbor)].drop(globals()['df_' + str(neighbor)].columns[[0]], axis=1)

            result_proba = lgbm.predict_proba(globals()['df_' + str(neighbor)])
            y_prob = result_proba[:, 1]

            result = (y_prob >= CUSTOM_THRESHOLD).astype(int)

            if (int(result[0]) == 0):
                result = 'Yes'
            else:
                result = 'No'    

            client_info.append({'clientId': neighbor, 
                                'repay' : result,
                                'score' : round(1000*result_proba[0][0]),
                                'probability0' : result_proba[0][0],
                                'probability1' : result_proba[0][1],
                                'threshold' : CUSTOM_THRESHOLD})

        repayment_status = {client['clientId']: client['repay'] for client in client_info}
        filtered_df = clients_to_predict[clients_to_predict['SK_ID_CURR'].isin(neighbors)]
        filtered_df['repay'] = filtered_df['SK_ID_CURR'].map(repayment_status)

        income_yes = filtered_df[filtered_df['repay'] == 'Yes']['AMT_INCOME_TOTAL'].tolist()
        income_no = filtered_df[filtered_df['repay'] == 'No']['AMT_INCOME_TOTAL'].tolist()

        result = {'Yes': income_yes, 'No': income_no}
        return result
    
@app.get('/api/clients/{id}/prediction/neighbors/score')
async def get_neighbors_score(id: int):
    ''' Endpoint to get the score of the neighbors of the current client
    '''
    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        client_idx = clients_to_predict_scaled.drop(columns=['SK_ID_CURR']).iloc[idx].values.reshape(1, -1)

        distances, indices = knn.kneighbors(client_idx)
        indices = [int(item) for item in indices[0]]
        indices = indices[1:]
        neighbors = clients_to_predict_scaled.iloc[indices]['SK_ID_CURR'].astype(int).tolist()
        client_info = []

        for neighbor in neighbors:
            globals()['df_' + str(neighbor)] = clients_to_predict[clients_to_predict['SK_ID_CURR'] == neighbor]
            globals()['df_' + str(neighbor)] = globals()['df_' + str(neighbor)].drop(globals()['df_' + str(neighbor)].columns[[0]], axis=1)

            result_proba = lgbm.predict_proba(globals()['df_' + str(neighbor)])
            y_prob = result_proba[:, 1]

            result = (y_prob >= CUSTOM_THRESHOLD).astype(int)

            if (int(result[0]) == 0):
                result = 'Yes'
            else:
                result = 'No'    

            client_info.append({'clientId': neighbor, 
                                'repay' : result,
                                'score' : round(1000*result_proba[0][0]),
                                'probability0' : result_proba[0][0],
                                'probability1' : result_proba[0][1],
                                'threshold' : CUSTOM_THRESHOLD})
            
        repayment_status = {client['clientId']: client['repay'] for client in client_info}
        scores = {client['clientId']: client['score'] for client in client_info}

        filtered_df = clients_to_predict[clients_to_predict['SK_ID_CURR'].isin(neighbors)]
        filtered_df['repay'] = filtered_df['SK_ID_CURR'].map(repayment_status)
        filtered_df['score'] = filtered_df['SK_ID_CURR'].map(scores)

        score_yes = filtered_df[filtered_df['repay'] == 'Yes']['score'].tolist()
        score_no = filtered_df[filtered_df['repay'] == 'No']['score'].tolist()

        result = {'Yes': score_yes, 'No': score_no}
        return result

@app.get('/api/clients/{id}/prediction/neighbors/amtCredit')
async def get_neighbors_credit_amount(id: int):
    ''' Endpoint to get the credit amount of the neighbors of the current client
    '''
    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        client_idx = clients_to_predict_scaled.drop(columns=['SK_ID_CURR']).iloc[idx].values.reshape(1, -1)

        distances, indices = knn.kneighbors(client_idx)
        indices = [int(item) for item in indices[0]]
        indices = indices[1:]
        neighbors = clients_to_predict_scaled.iloc[indices]['SK_ID_CURR'].astype(int).tolist()
        client_info = []

        for neighbor in neighbors:
            globals()['df_' + str(neighbor)] = clients_to_predict[clients_to_predict['SK_ID_CURR'] == neighbor]
            globals()['df_' + str(neighbor)] = globals()['df_' + str(neighbor)].drop(globals()['df_' + str(neighbor)].columns[[0]], axis=1)

            result_proba = lgbm.predict_proba(globals()['df_' + str(neighbor)])
            y_prob = result_proba[:, 1]

            result = (y_prob >= CUSTOM_THRESHOLD).astype(int)

            if (int(result[0]) == 0):
                result = 'Yes'
            else:
                result = 'No'    

            client_info.append({'clientId': neighbor, 
                                'repay' : result,
                                'score' : round(1000*result_proba[0][0]),
                                'probability0' : result_proba[0][0],
                                'probability1' : result_proba[0][1],
                                'threshold' : CUSTOM_THRESHOLD})

        repayment_status = {client['clientId']: client['repay'] for client in client_info}
        filtered_df = clients_to_predict[clients_to_predict['SK_ID_CURR'].isin(neighbors)]
        filtered_df['repay'] = filtered_df['SK_ID_CURR'].map(repayment_status)

        income_yes = filtered_df[filtered_df['repay'] == 'Yes']['AMT_CREDIT'].tolist()
        income_no = filtered_df[filtered_df['repay'] == 'No']['AMT_CREDIT'].tolist()

        result = {'Yes': income_yes, 'No': income_no}
        return result
    
@app.get('/api/clients/{id}/prediction/neighbors/loanLength')
async def get_neighbors_credit_amount(id: int):
    ''' Endpoint to get the duration of the loan of the neighbors of the current client
    '''
    clients_id = clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(clients_to_predict[clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        client_idx = clients_to_predict_scaled.drop(columns=['SK_ID_CURR']).iloc[idx].values.reshape(1, -1)

        distances, indices = knn.kneighbors(client_idx)
        indices = [int(item) for item in indices[0]]
        indices = indices[1:]
        neighbors = clients_to_predict_scaled.iloc[indices]['SK_ID_CURR'].astype(int).tolist()
        client_info = []

        for neighbor in neighbors:
            globals()['df_' + str(neighbor)] = clients_to_predict[clients_to_predict['SK_ID_CURR'] == neighbor]
            globals()['df_' + str(neighbor)] = globals()['df_' + str(neighbor)].drop(globals()['df_' + str(neighbor)].columns[[0]], axis=1)

            result_proba = lgbm.predict_proba(globals()['df_' + str(neighbor)])
            y_prob = result_proba[:, 1]

            result = (y_prob >= CUSTOM_THRESHOLD).astype(int)

            if (int(result[0]) == 0):
                result = 'Yes'
            else:
                result = 'No'    

            client_info.append({'clientId': neighbor, 
                                'repay' : result,
                                'score' : round(1000*result_proba[0][0]),
                                'probability0' : result_proba[0][0],
                                'probability1' : result_proba[0][1],
                                'threshold' : CUSTOM_THRESHOLD})

        repayment_status = {client['clientId']: client['repay'] for client in client_info}
        filtered_df = clients_to_predict[clients_to_predict['SK_ID_CURR'].isin(neighbors)]
        filtered_df['repay'] = filtered_df['SK_ID_CURR'].map(repayment_status)

        filtered_df['loanLength'] = 12*(filtered_df['AMT_CREDIT'].astype(float) / filtered_df['AMT_ANNUITY'].astype(float))

        income_yes = filtered_df[filtered_df['repay'] == 'Yes']['loanLength'].tolist()
        income_no = filtered_df[filtered_df['repay'] == 'No']['loanLength'].tolist()

        result = {'Yes': income_yes, 'No': income_no}
        return result

# Endpoints to get information about clients already present in the database
@app.get('/api/statistics/loans')
async def get_stats_loan():

    count_0 = current_clients[current_clients['TARGET'] == 0]['SK_ID_CURR'].count()
    count_1 = current_clients[current_clients['TARGET'] == 1]['SK_ID_CURR'].count()

    loans = {
        'repaid': int(count_0), 
        'defaulted': int(count_1)
    }

    return loans

@app.get('/api/statistics/genders')
async def get_stats_gender():

    df_gender = current_clients[['SK_ID_CURR', 'CODE_GENDER', 'TARGET']].fillna(0)
    df_gender['CODE_GENDER'] = df_gender['CODE_GENDER'].replace({0: 'M', 1: 'F'})

    result = {}
    for index, row in df_gender.iterrows():
        result[row['SK_ID_CURR']] = [row['CODE_GENDER'], 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/ages')
async def get_stats_ages():

    df_ages = current_clients[['SK_ID_CURR', 'DAYS_BIRTH', 'TARGET']].fillna(0)
    df_ages['age'] = round(df_ages['DAYS_BIRTH']/-365)
    df_ages.drop(columns=['DAYS_BIRTH'])

    result = {}
    for index, row in df_ages.iterrows():
        result[row['SK_ID_CURR']] = [row['age'], 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/total_incomes')
async def get_statistics_total_income():

    df_income = current_clients[['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'TARGET']].fillna(0)

    result = {}
    for index, row in df_income.iterrows():
        result[row['SK_ID_CURR']] = [row['AMT_INCOME_TOTAL'], 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/credits')
async def get_statistics_credit():

    df_credit = current_clients[['SK_ID_CURR', 'AMT_CREDIT', 'TARGET']].fillna(0)

    result = {}
    for index, row in df_credit.iterrows():
        result[row['SK_ID_CURR']] = [row['AMT_CREDIT'], 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/annuity')
async def get_statistics_annuity():

    df_credit = current_clients[['SK_ID_CURR', 'AMT_ANNUITY', 'TARGET']].fillna(0)

    result = {}
    for index, row in df_credit.iterrows():
        result[row['SK_ID_CURR']] = [row['AMT_ANNUITY'], 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/length_loan')
async def get_statistics_length_loan():

    df_credit = current_clients[['SK_ID_CURR', 'AMT_CREDIT', 'AMT_ANNUITY', 'TARGET']].fillna(0)
    #df_credit['length'] = round(12*(df_credit['AMT_CREDIT'] / df_credit['AMT_ANNUITY']))

    result = {}
    for index, row in df_credit.iterrows():
        result[row['SK_ID_CURR']] = [row['AMT_CREDIT'], row['AMT_ANNUITY'], 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/payment_rate')
async def get_statistics_payment_rate():

    df_credit = current_clients[['SK_ID_CURR', 'PAYMENT_RATE', 'TARGET']].fillna(0)

    result = {}
    for index, row in df_credit.iterrows():
        result[row['SK_ID_CURR']] = [round(100*row['PAYMENT_RATE'], 2), 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result

@app.get('/api/statistics/credit_income_percent')
async def get_statistics_credit_income_percent():

    df_credit = current_clients[['SK_ID_CURR', 'CREDIT_INCOME_PERCENT', 'TARGET']].fillna(0)

    result = {}
    for index, row in df_credit.iterrows():
        result[row['SK_ID_CURR']] = [round(row['CREDIT_INCOME_PERCENT'], 2), 'repaid' if row['TARGET'] == 0 else 'defaulted']

    return result