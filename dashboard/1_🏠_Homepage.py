import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.figure_factory as ff

API_ADDRESS = 'http://' + str(os.environ['AWS_PUBLIC_IP_ADDRESS_API'])

# API
@st.cache_data  
def get_genders():
    response = requests.get(API_ADDRESS + '/api/statistics/genders')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None
@st.cache_data      
def get_ages():
    response = requests.get(API_ADDRESS + '/api/statistics/ages')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None
@st.cache_data      
def get_loan():
    response = requests.get(API_ADDRESS + '/api/statistics/loans')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None
@st.cache_data      
def get_incomes():
    response = requests.get(API_ADDRESS + '/api/statistics/total_incomes')
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
@st.cache_data      
def get_credits():
    response = requests.get(API_ADDRESS + '/api/statistics/credits')
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
@st.cache_data  
def get_length_loan():
    response = requests.get(API_ADDRESS + '/api/statistics/length_loan')
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
@st.cache_data      
def get_payment_rate():
    response = requests.get(API_ADDRESS + '/api/statistics/payment_rate')
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

# Plotting functions
@st.cache_data  
def plot_gender(data: dict):

    categories = {
        ('M', 'repaid'): 0, 
        ('M', 'defaulted'): 0, 
        ('F', 'repaid'): 0, 
        ('F', 'defaulted'): 0
    }

    for key, value in data.items():
        gender, status = value
        categories[(gender, status)] += 1

    gender_status = [('M', 'repaid'), ('M', 'defaulted'), ('F', 'repaid'), ('F', 'defaulted')]
    counts = [categories[cat] for cat in gender_status]
    colors = ['skyblue', 'lightcoral', 'skyblue', 'lightcoral']

    bar_width = 0.35
    index = range(len(gender_status))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(index, counts, bar_width, label='Counts', color=colors)

    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(['Male', 'Female'])

    legend_labels = ['Repaid', 'Defaulted']
    legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in [0, 1]]
    ax.legend([legend_handles[i] for i in [0, 1]], legend_labels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

@st.cache_data  
def plot_ages(data: dict):

    fig, ax = plt.subplots()
    ax.hist(data.values(), bins=10)
    ax.set_xlabel('Ages')
    ax.set_ylabel('Number of clients')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    st.pyplot(fig, use_container_width=True)

@st.cache_data  
def plot_loan(data: dict):

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['skyblue', 'lightcoral']

    ax.pie(data.values(), labels=['Repaid', 'Defaulted'], autopct='%1.1f%%', radius=1, startangle=90, textprops={'fontsize': 12}, colors=colors)
    inner_circle = plt.Circle((0, 0), 0.4, color='white')
    ax.add_artist(inner_circle)

    ax.axis('equal')

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

@st.cache_data  
def plot_total_incomes(data: dict):

    repaid_incomes = []
    defaulted_incomes = []

    for client_id, [income, status] in data.items():
        if status == 'repaid':
            repaid_incomes.append(income)
        elif status == 'defaulted':
            defaulted_incomes.append(income)

    hist_data = [repaid_incomes, defaulted_incomes]
    labels = ['repaid', 'defaulted']

    fig = ff.create_distplot(hist_data, labels, bin_size=[0.25, 0.25])

    st.plotly_chart(fig, use_container_width=True)

@st.cache_data  
def plot_total_length_loan(data: dict):

    length_defaulted = []
    length_repaid = []

    for value_list in data.values():
        credit = value_list[0]
        annuity = value_list[1]
        target = value_list[2]

        if target == 'defaulted': 
            if annuity != 0:
                length_defaulted.append(int(12*credit / annuity))
        elif annuity != 'repaid':
            if annuity != 0:
                length_repaid.append(int(12*credit / annuity))

    mean_val_defaulted, mean_val_repaid = np.mean(length_defaulted), np.mean(length_repaid)
    median_val_defaulted, median_val_repaid = np.median(length_defaulted), np.median(length_repaid)
    max_val_defaulted, max_val_repaid = np.max(length_defaulted), np.max(length_repaid)
    min_val_defaulted, min_val_repaid = np.min(length_defaulted), np.min(length_repaid)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].hist(length_defaulted, bins=20, color='lightcoral', alpha=0.7)
    axs[1].hist(length_repaid, bins=20, color='skyblue', alpha=0.7)

    axs[0].text(0.05, 0.9, 'Defaulted', transform=axs[0].transAxes, fontsize=10, color='red')
    axs[1].text(0.05, 0.9, 'Repaid', transform=axs[1].transAxes, fontsize=10, color='blue')

    axs[0].axvline(mean_val_defaulted, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {round(mean_val_defaulted)}')
    axs[0].axvline(median_val_defaulted, color='green', linestyle='dashed', linewidth=1, label=f'Median: {round(median_val_defaulted)}')
    axs[0].axvline(max_val_defaulted, color='orange', linestyle='dashed', linewidth=1, label=f'Max: {round(max_val_defaulted)}')
    axs[0].axvline(min_val_defaulted, color='purple', linestyle='dashed', linewidth=1, label=f'Min: {round(min_val_defaulted)}')
    
    axs[1].axvline(mean_val_repaid, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {round(mean_val_repaid)}')
    axs[1].axvline(median_val_repaid, color='green', linestyle='dashed', linewidth=1, label=f'Median: {round(median_val_repaid)}')
    axs[1].axvline(max_val_repaid, color='orange', linestyle='dashed', linewidth=1, label=f'Max: {round(max_val_repaid)}')
    axs[1].axvline(min_val_repaid, color='purple', linestyle='dashed', linewidth=1, label=f'Min: {round(min_val_repaid)}')
    
    for i in [0, 1]: 
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

@st.cache_data  
def plot_total_payment_rates(data: dict):

    rate_defaulted = [value[0] for value in data.values() if value[1] == 'defaulted']
    rate_repaid = [value[0] for value in data.values() if value[1] == 'repaid']

    mean_val_defaulted, mean_val_repaid = np.mean(rate_defaulted), np.mean(rate_repaid)
    median_val_defaulted, median_val_repaid = np.median(rate_defaulted), np.median(rate_repaid)
    max_val_defaulted, max_val_repaid = np.max(rate_defaulted), np.max(rate_repaid)
    min_val_defaulted, min_val_repaid = np.min(rate_defaulted), np.min(rate_repaid)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].hist(rate_defaulted, bins=20, color='lightcoral', alpha=0.7)
    axs[1].hist(rate_repaid, bins=20, color='skyblue', alpha=0.7)

    axs[0].text(0.05, 0.9, 'Defaulted', transform=axs[0].transAxes, fontsize=10, color='red')
    axs[1].text(0.05, 0.9, 'Repaid', transform=axs[1].transAxes, fontsize=10, color='blue')

    axs[0].axvline(mean_val_defaulted, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {round(mean_val_defaulted, 2)}%')
    axs[0].axvline(median_val_defaulted, color='green', linestyle='dashed', linewidth=1, label=f'Median: {round(median_val_defaulted, 2)}%')
    axs[0].axvline(max_val_defaulted, color='orange', linestyle='dashed', linewidth=1, label=f'Max: {round(max_val_defaulted, 2)}%')
    axs[0].axvline(min_val_defaulted, color='purple', linestyle='dashed', linewidth=1, label=f'Min: {round(min_val_defaulted, 2)}%')
    
    axs[1].axvline(mean_val_repaid, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {round(mean_val_repaid, 2)}%')
    axs[1].axvline(median_val_repaid, color='green', linestyle='dashed', linewidth=1, label=f'Median: {round(median_val_repaid, 2)}%')
    axs[1].axvline(max_val_repaid, color='orange', linestyle='dashed', linewidth=1, label=f'Max: {round(max_val_repaid, 2)}%')
    axs[1].axvline(min_val_repaid, color='purple', linestyle='dashed', linewidth=1, label=f'Min: {round(min_val_repaid, 2)}%')
    
    for i in [0, 1]: 
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()

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
    st.markdown('**Distribution of loan repayment**')
    info = get_loan()
    plot_loan(info)
    st.markdown('\n')

    st.markdown('**Distribution of annual incomes:**')
    info = get_incomes()
    plot_total_incomes(info)
    st.markdown('\n')

    st.markdown('**Distribution of length loan in months:**')
    info = get_length_loan()
    plot_total_length_loan(info)
    st.markdown('\n')

with col2:
    st.markdown('**Distribution of gender**')
    info = get_genders()
    plot_gender(info)
    st.markdown('\n')

    st.markdown('**Distribution of credit values:**')
    info = get_credits()
    plot_total_incomes(info)
    st.markdown('\n')

    st.markdown('**Distribution of payment rates:**')
    info = get_payment_rate()
    plot_total_payment_rates(info)
    st.markdown('\n')

