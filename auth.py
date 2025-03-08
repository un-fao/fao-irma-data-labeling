#
#  User Authentication
#
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import google.cloud.secretmanager as secretmanager
import streamlit_authenticator as stauth

@st.cache_data
def load_auth_config():
    if st.session_state.debug:
        print('load_auth_config')
    if st.session_state.use_st_secrets:
        auth_config = yaml.load(st.secrets['app_authentication']['app_auth_settings'], Loader=SafeLoader)
    else:
        secret_client = secretmanager.SecretManagerServiceClient()
        secret_name = "projects/second-impact-342800/secrets/irma_app_auth_settings/versions/latest"
        response = secret_client.access_secret_version(request={"name": secret_name})
        secret = response.payload.data.decode("UTF-8")
        auth_config = yaml.load(secret, Loader=SafeLoader)
    return auth_config


def authenticate(page_name):
    auth_config = load_auth_config()
    authenticator = stauth.Authenticate(
        auth_config['credentials'],
        auth_config['cookie']['name'],
        auth_config['cookie']['key'],
        auth_config['cookie']['expiry_days'],
        key=f'{page_name}_login'
    )
    st.session_state.authenticator = authenticator
    authenticator.login(key=f'{page_name}_login')
