import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

# API
def get_clients():
    response = requests.get('http://127.0.0.1:8000/api/clients')
    if response.status_code == 200: 
        data = response.json()
        clients = data['clientsID']
        return clients 
    else:
        st.error('Failed to get clients')
        return None
    
def get_client_personal_information(id):
    response = requests.get(f'http://127.0.0.1:8000/api/clients/{id}/personal_information')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
def get_client_bank_information(id):
    response = requests.get(f'http://127.0.0.1:8000/api/clients/{id}/bank_information')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
def get_client_prediction(id):
    response = requests.get(f'http://127.0.0.1:8000/api/clients/{id}/prediction')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
def get_client_shap_values(id):
    response = requests.get(f'http://127.0.0.1:8000/api/clients/{id}/prediction/shap/local')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
def get_model_shap_values(id):
    response = requests.get(f'http://127.0.0.1:8000/api/clients/{id}/prediction/shap/global')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

def plot_score(value): 
    colors = ['green', 'green', 'yellow', 'yellow', 'yellow', 'yellow', 'red', 'red', 'red', 'red']
    values = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

    x_axis_vals = [0, 0.314, 0.628, 0.942, 1.256, 1.57, 1.884, 2.198, 2.512, 2.826]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='polar')
    ax.set_thetamin(180)
    ax.set_thetamax(0)

    ax.bar(x=x_axis_vals, width=0.5, height=0.5, bottom=2, linewidth=3, edgecolor='white', color=colors, align='edge')

    for loc, val in zip([0, 0.314, 0.628, 0.942, 1.256, 1.57, 1.884, 2.198, 2.512, 2.826, 3.14], values):
        plt.annotate(val, xy=(loc, 2.5), ha='right' if val < 600 else 'left')

    position = 3.14 - (value/1000)*3.14

    plt.annotate(str(value), xytext=(0, 0), xy=(position, 2.0),
                arrowprops=dict(arrowstyle='wedge, tail_width=0.4', color='black', shrinkA=0),
                bbox=dict(boxstyle='circle', facecolor='black', linewidth=2.0),
                fontsize=25, color='white', ha='center'
                )

    ax.set_axis_off()
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

def plot_likelihood(repay, default, threshold):
    colors = ['green', 'red']
    data = {
        'Likelihood': ['To repay', 'To default'], 
        'Probability': [repay, default]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df['Likelihood'], df['Probability'], color=colors)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_ylim(0, 1)
    ax.axhline(y=threshold, color='darkred', linestyle='--', linewidth=1)

    plt.xticks(ha='center')

    st.pyplot(fig)

def plot_extsources(ext1, ext2, ext3):
    colors = ['orange', 'purple', 'blue']
    data = {
        'External source': ['1', '2', '3'], 
        'Values': [ext1, ext2, ext3]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df['External source'], df['Values'], color=colors)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_ylim(0, 1)

    plt.xticks(ha='center')

    st.pyplot(fig)

st.set_page_config(
    page_title='Prêt à dépenser - Default Risk - Client',
    page_icon='💳'
)

st.title('Client information')

clients = get_clients()
if clients: 
    selected_info = st.selectbox('Select client', clients)


tab1, tab2,tab3, tab4, tab5 = st.tabs(['🆔 Personal information', 
                                       '🏦 Bank information', 
                                       '🎯 Prediction', 
                                       '📊 Local analysis', 
                                       '🌍 Global analysis'])

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

        col1, col2= st.columns(2)

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
            st.markdown('**External sources:**')
            plot_extsources(info['extSource1'], info['extSource2'], info['extSource3'])

with tab3:
    if selected_info:
        info = get_client_prediction(selected_info)

        col1, col2 = st.columns(2)

        with col1: 
            st.markdown('**Credit score:**')
            plot_score(info['score'])
            st.markdown('**Threshold of selection:**')
            st.markdown('Probability of defaulting superior to ' + str(info['threshold']))

        with col2: 
            st.markdown('**Likelihood:**')
            plot_likelihood(info['probability0'], info['probability1'], info['threshold'])
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('**Should the loan request be accepted:**')
            st.markdown(info['repay'])


with tab4:
    if selected_info:
        info = get_client_shap_values(selected_info)
        st.write(info)

with tab5:
    if selected_info:
        info = get_model_shap_values(selected_info)
        st.write(info)