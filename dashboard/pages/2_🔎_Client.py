import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns 
import matplotlib.ticker as mtick
import os
import plotly.figure_factory as ff
import plotly.express as px

API_ADDRESS = 'http://' + str(os.environ['AWS_PUBLIC_IP_ADDRESS_API'])

# API
@st.cache_data  
def get_clients():
    response = requests.get(API_ADDRESS + '/api/clients')
    if response.status_code == 200: 
        data = response.json()
        clients = data['clientsID']
        return clients 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def get_client_personal_information(id):
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/personal_information')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def get_client_bank_information(id):
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/bank_information')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def get_client_prediction(id):
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def get_client_shap_values(id):
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/shap/local')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data  
def get_model_shap_values(id):
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/shap/global')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def get_neighbors(id):
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/neighbors')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def get_neighbors_total_income(id): 
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/neighbors/totalIncome')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data  
def get_neighbors_score(id): 
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/neighbors/score')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
@st.cache_data 
def get_neighbors_credit_amount(id): 
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/neighbors/amtCredit')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
@st.cache_data 
def get_neighbors_loan_duration(id): 
    response = requests.get(API_ADDRESS + f'/api/clients/{id}/prediction/neighbors/loanLength')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

# Plot functions
def plot_score(value): 
    if value < 400:
        color = 'red'
    elif value < 800:
        color = 'yellow'
    else:
        color = 'green'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': color}}
    ))

    st.plotly_chart(fig, use_container_width=True)

def plot_likelihood(repay, default, threshold):
    data = {
        'Likelihood': ['To repay', 'To default'], 
        'Probability (%)': [round(100*repay, 2), round(100*default, 2)]
    }
    df = pd.DataFrame(data)
    df = df.set_index('Likelihood')

    st.bar_chart(df, use_container_width=True)

def plot_extsources(ext1, ext2, ext3):
    data = {
        'External source': ['1', '2', '3'], 
        'Values': [ext1, ext2, ext3]
    }
    df = pd.DataFrame(data)
    df = df.set_index('External source')

    st.bar_chart(df, use_container_width=True)

def plot_shap(data: dict):

    df = pd.DataFrame({'Features': data.keys(), 'Importance': data.values()})
    df['Color'] = df['Importance'].apply(lambda x: 'positive' if x >= 0 else 'negative')

    colors = {'positive': 'blue', 'negative': 'red'}

    fig = px.bar(df, x='Importance', y='Features', color='Color', color_discrete_map=colors)
    fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

def plot_neighbors_scores(data):
    fig_yes = px.histogram(x=data['Yes'], labels={'x': 'Scores'}, color_discrete_sequence=['green'])
    fig_no = px.histogram(x=data['No'], labels={'x': 'Scores'}, color_discrete_sequence=['red'])

    tab1, tab2 = st.tabs(['Likely to repay', 'Likely to default'])
    with tab1: 
        st.plotly_chart(fig_yes, use_container_width=True)
    with tab2:
        st.plotly_chart(fig_no, use_container_width=True)

def plot_neighbors_annual_income(data):
    fig_yes = px.histogram(x=data['Yes'], labels={'x': 'Total income'}, color_discrete_sequence=['green'])
    fig_no = px.histogram(x=data['No'], labels={'x': 'Total income'}, color_discrete_sequence=['red'])

    tab1, tab2 = st.tabs(['Likely to repay', 'Likely to default'])
    with tab1: 
        st.plotly_chart(fig_yes, use_container_width=True)
    with tab2:
        st.plotly_chart(fig_no, use_container_width=True)

def plot_neighbors_credit_amount(data):
    fig_yes = px.histogram(x=data['Yes'], labels={'x': 'Credit amount'}, color_discrete_sequence=['green'])
    fig_no = px.histogram(x=data['No'], labels={'x': 'Credit amount'}, color_discrete_sequence=['red'])

    tab1, tab2 = st.tabs(['Likely to repay', 'Likely to default'])
    with tab1: 
        st.plotly_chart(fig_yes, use_container_width=True)
    with tab2:
        st.plotly_chart(fig_no, use_container_width=True)

def plot_neighbors_loan_duration(data):
    fig_yes = px.histogram(x=data['Yes'], labels={'x': 'Loan duration'}, color_discrete_sequence=['green'])
    fig_no = px.histogram(x=data['No'], labels={'x': 'Loan duration'}, color_discrete_sequence=['red'])

    tab1, tab2 = st.tabs(['Likely to repay', 'Likely to default'])
    with tab1: 
        st.plotly_chart(fig_yes, use_container_width=True)
    with tab2:
        st.plotly_chart(fig_no, use_container_width=True)

st.set_page_config(
    page_title='Prêt à dépenser - Default Risk - Client',
    page_icon='💳'
)

st.markdown("<h2 style='text-align: center;'>Client information</h2>", unsafe_allow_html=True)
st.markdown('')

clients = get_clients()
if clients: 
    selected_info = st.selectbox('Select client', clients)
    st.markdown('')

tab1, tab2,tab3, tab4, tab5, tab6 = st.tabs(['🆔 Personal information', 
                                            '🏦 Financial information', 
                                            '🎯 Prediction', 
                                            '💼 Similar clients', 
                                            '📊 Local analysis', 
                                            '🌍 Global analysis'
                                            ])

with tab1:
    if selected_info:
        info = get_client_personal_information(selected_info)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('\n\n')
            st.markdown('**Gender:**')
            st.markdown('\n\n')
            st.markdown('**Children:**')
            st.markdown('\n\n')
            st.markdown('**Age:**')
            st.markdown('\n\n')
            st.markdown('**Birthday:**')

        with col3:
            st.markdown('\n\n')
            st.markdown('**Car:**')
            st.markdown('\n\n')
            st.markdown('**Realty:**')
            st.markdown('\n\n')
            st.markdown('**Civil status:**')
            st.markdown('\n\n')
            st.markdown('**Education type:**')

        with col2: 
            st.markdown('\n\n')
            st.markdown(info['gender'])
            st.markdown('\n\n')
            st.markdown(info['countChildren'])
            st.markdown('\n\n')
            st.markdown(info['age'])
            st.markdown('\n\n')
            st.markdown(info['birthday'])

        with col4:
            st.markdown('\n\n')
            st.markdown(info['ownCar'])
            st.markdown('\n\n')
            st.markdown(info['ownRealty'])
            st.markdown('\n\n')
            st.markdown(info['civilStatus'])
            st.markdown('\n\n')
            st.markdown(info['educationType'])

with tab2:
    if selected_info:
        info = get_client_bank_information(selected_info)

        col1, col2 = st.columns(2)

        with col1: 
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.markdown('\n\n')
                st.markdown('**Annual income:**')
                st.markdown('\n\n')
                st.markdown('**Seniority:**')
                st.markdown('\n\n')
                st.markdown('**Client since:**')
                st.markdown('\n\n')
                st.markdown('**Income type:**')
                st.markdown('\n\n')
                st.markdown('**Loan amount:**')
                st.markdown('\n\n')
                st.markdown('**Loan duration:**')
                st.markdown('\n\n')
                st.markdown('**Loan annuity:**')
                st.markdown('\n\n')
                st.markdown('**Payment rate:**')
                st.markdown('\n\n')
                st.markdown('**Loan / income ratio:**')

            with col1_2: 
                st.markdown('\n\n')
                st.markdown('$ ' + str(info['totalIncome']))
                st.markdown('\n\n')
                st.markdown(str(info['seniority']) + ' years' if info['seniority']>1 else (str(info['seniority']) + ' year') if info['seniority'] == 1 else '')
                st.markdown('\n\n')
                st.markdown(info['registrationSince'])
                st.markdown('\n\n')
                st.markdown(info['incomeType'])
                st.markdown('\n\n')
                st.markdown('$ ' + str(info['amtCredit']))
                st.markdown('\n\n')
                st.markdown(str(info['lengthCredit']) + ' months')
                st.markdown('\n\n')
                st.markdown('$ ' + str(info['annualCredit']))
                st.markdown('\n\n')
                st.markdown(str(info['paymentRate']) + '%')
                st.markdown('\n\n')
                st.markdown(info['creditIncomeRatio'])


        with col2: 
            st.markdown('\n\n')
            st.markdown('**Normalized score from other financial institutions:**')
            plot_extsources(info['extSource1'], info['extSource2'], info['extSource3'])

with tab3:
    if selected_info:
        info = get_client_prediction(selected_info)

        col1, col2 = st.columns(2)

        with col1: 
            st.markdown('**Credit score:**')
            plot_score(info['score'])
            st.markdown('**Threshold of selection:**')
            st.markdown('Loan request should not be accepted if the probability of defaulting superior is to ' + str(round(100*info['threshold'], 2)) + '%')

        with col2: 
            st.markdown('**Likelihood:**')
            plot_likelihood(info['probability0'], info['probability1'], info['threshold'])
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('**Should the loan request be accepted:**')
            st.markdown(info['repay'])

with tab4:
    if selected_info:
        col1, col2 = st.columns(2)
        with col1: 
            st.markdown('**Distribution of annual income:**')
            data = get_neighbors_total_income(selected_info)
            plot_neighbors_annual_income(data)
            st.markdown('\n')

            st.markdown('**Distribution of credit amount:**')
            data = get_neighbors_credit_amount(selected_info)
            plot_neighbors_credit_amount(data)

        with col2:
            st.markdown('**Distribution of credit score:**')
            data = get_neighbors_score(selected_info)
            plot_neighbors_scores(data)
            st.markdown('\n')

            st.markdown('**Distribution of loan duration in months:**')
            data = get_neighbors_loan_duration(selected_info)
            plot_neighbors_loan_duration(data)

with tab5:
    if selected_info:
        data = get_client_shap_values(selected_info)
        plot_shap(data)

with tab6:
    if selected_info:
        info = get_model_shap_values(selected_info)
        plot_shap(info)