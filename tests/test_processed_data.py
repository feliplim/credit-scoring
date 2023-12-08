import pandas as pd
import pytest
import boto3
import os

@pytest.fixture(scope='module')
def s3_client():
    '''Create an S3 client'''
    region_name = os.getenv('AWS_REGION')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    return boto3.client('s3', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

@pytest.fixture(scope='module')
def get_train_data(s3_client):
    '''Get customers processed train data to feed into the tests'''
    bucket_name = 'credit-scoring-openclassrooms'
    file = 'data/processed/train_feature_engineering_encoded.csv'
    obj = s3_client.get_object(Bucket=bucket_name, Key=file)
    return pd.read_csv(obj['Body'])

@pytest.fixture(scope='module')
def get_test_data(s3_client):
    '''Get customers processed test data to feed into the tests'''
    bucket_name = 'credit-scoring-openclassrooms'
    file = 'data/processed/test_feature_engineering_encoded.csv'
    obj = s3_client.get_object(Bucket=bucket_name, Key=file)
    return pd.read_csv(obj['Body'])

def test_train_duplicates(get_train_data):
    '''Check if the train duplicated dataframe is empty --> no duplicates'''
    train_data = get_train_data
    duplicates = train_data[train_data.duplicated()]
    assert duplicates.empty

def test_test_duplicates(get_test_data):
    '''Check if the test duplicated dataframe is empty --> no duplicates'''
    test_data = get_test_data
    duplicates = test_data[test_data.duplicated()]
    assert duplicates.empty

def test_train_target_col(get_train_data):
    '''Test that the train dataframe has a 'target' column'''
    train_data = get_train_data
    assert 'TARGET' in train_data.columns

def test_train_test_sizes(get_train_data, get_test_data):
    '''Check that train and test dataframe have the same columns (but target)'''
    train_data = get_train_data
    test_data = get_test_data
    train_size = train_data.drop(columns='TARGET').shape[1]
    test_size = test_data.shape[1]
    assert train_size == test_size