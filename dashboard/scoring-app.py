import streamlit as st
import requests

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
    
def get_client_information(id):
    response = requests.get(f'http://127.0.0.1:8000/api/clients/{id}')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None
    
# Main page
st.title('Prêt à dépenser - Default Risk')

# Sidebar
st.sidebar.header('Menu')
selection = st.sidebar.radio('Go to', ['Homepage', 'Global analysis', 'Client\'s profile'])

# Display content based on the selection
if selection == 'Homepage':
    st.write('\n')
    st.write('This dashboard was created to optimize your decision when dealing with a client.')
    st.write('\n')
    st.write('Go to "Global analysis" to check all clients\' behavior and the typical profile of clients having repaid loans and the ones having problems to repay it.')
    st.write('\n')
    st.write('Go to "Client\'s profile" to check a particular client behavior and the score for his/her loan request.')

elif selection == 'Global analysis':
    st.write('\n')
    st.write('### GLOBAL ANALYSIS')


elif selection == 'Client\'s profile':
    st.write('\n')
    st.write('### CLIENT\'S PROFILE')

    col1, col2, col3, col4 = st.columns(4)

    if col1.button('C'):
        st.write('It will show risk score for the client')

    if col2.button('P'):
        st.write('It will show personal information')

    if col3.button('F'):
        st.write('It will show financial information')

    if col4.button('O'):
        st.write('It will show comparation with other clients')

    clients = get_clients()
    if clients: 
        selected_info = st.selectbox("Select client", clients)

        if selected_info:
            info = get_client_information(selected_info)
            st.write(info)

    