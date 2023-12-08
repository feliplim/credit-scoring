import pandas as pd
import pytest
import os

# Set local directory
project_path = '/Users/felipelima/Documents/projets/credit-scoring/'
os.chdir(project_path)

@pytest.fixture(scope='module')
def get_train_data():
    '''Get customers processed train data to feed into the tests'''
    return pd.read_csv('data/processed/train_feature_engineering_encoded.csv')

@pytest.fixture(scope='module')
def get_test_data():
    '''Get customers processed test data to feed into the tests'''
    return pd.read_csv('data/processed/test_feature_engineering_encoded.csv')

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