import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

API_ADRESS = 'http://127.0.0.1:8000'

# API
def get_genders():
    response = requests.get(API_ADRESS + '/api/statistics/genders')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None
    
def get_ages():
    response = requests.get(API_ADRESS + '/api/statistics/ages')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None
    
def get_loan():
    response = requests.get(API_ADRESS + '/api/statistics/loans')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None


def plot_gender(data: dict):

    fig, ax = plt.subplots()

    repaid_men = data['repaidMen']
    defaulted_men = data['defaultedMen']
    repaid_women = data['repaidWomen']
    defaulted_women = data['defaultedWomen']

    genders = ['men', 'women']
    loans = {
        'repaid': [repaid_men, repaid_women], 
        'defaulted': [defaulted_men, defaulted_women]
    }

    x = np.arange(len(genders))  # the label locations  
    width = 0.25  # the width of the bars
    multiplier = 0

    for loan, amount in loans.items(): 
        offset = width * multiplier
        rects = ax.bar(x + offset, amount, width, label=loan)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Number of clients')
    ax.set_xticks(x + width, genders)
    ax.legend(loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

def plot_hist(data: dict):

    fig, ax = plt.subplots()
    ax.hist(data.values(), bins=10)
    ax.set_xlabel('Ages')
    ax.set_ylabel('Number of clients')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    st.pyplot(fig, use_container_width=True)

def plot_loan(data: dict):

    fig, ax = plt.subplots()

    ax.pie(data.values(), labels=['Repaid', 'Defaulted'], autopct='%1.1f%%', radius=1, startangle=90, textprops={'fontsize': 12})
    inner_circle = plt.Circle((0, 0), 0.4, color='white')
    ax.add_artist(inner_circle)

    ax.axis('equal')

    st.pyplot(fig, use_container_width=True)


st.set_page_config(
    page_title='Prêt à dépenser - Default Risk - Homepage',
    page_icon='💳'
)

st.markdown("<h2 style='text-align: center;'>General statistics about our current clients</h2>", unsafe_allow_html=True)
st.markdown('')
st.markdown('')

col1, col2 = st.columns(2)

with col1:
    st.markdown('**Loan repayment**')
    info = get_loan()
    plot_loan(info)

with col2:
    st.markdown('**Gender**')
    info = get_genders()
    plot_gender(info)

