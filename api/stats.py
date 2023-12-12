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
              description='Get current clients stats', 
              version='0.1.0')

# Get dataframes
current_clients = read('../data/processed/train_feature_engineering_encoded_extract.csv.gz')

@app.get('/')
async def read_root():
    return 'Home Credit Default Stats API'

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