#  GEE Authentication
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ee
from google.oauth2 import service_account
import streamlit as st
import google.cloud.secretmanager as secretmanager
import toml


def get_gee_credentials():
    if st.session_state.debug:
        print('get_gee_credentials')
    if st.session_state.use_st_secrets:
        config = st.secrets['gee_authentication']
    else:
        secret_client = secretmanager.SecretManagerServiceClient()
        secret_name = "projects/second-impact-342800/secrets/irma_gee_connection/versions/latest"
        response = secret_client.access_secret_version(request={"name": secret_name})
        secret = response.payload.data.decode("UTF-8")
        config = toml.loads(secret)['gee_authentication']
    return config


def check_gee_connection(retries=0):
    if 'gee_auth_expires_by' not in st.session_state or datetime.now() >= st.session_state.gee_auth_expires_by:
        if st.session_state.debug:
            print('authenticating')
        try:
            gee_config = get_gee_credentials()
            gee_credentials = service_account.Credentials.from_service_account_info(gee_config, scopes=ee.oauth.SCOPES)
            ee.Initialize(gee_credentials)
            if st.session_state.debug:
                print('authenticated')
            st.session_state.gee_auth_expires_by = datetime.now() + relativedelta(hours=1)
        except Exception as e:
            if retries < 5:
                if st.session_state.debug:
                    print(f'authentication failed {e}, retry', retries+1)
                check_gee_connection(retries+1)
            else:
                st.error('GEE authentication error.  Try refreshing the browser window.')
                st.stop()
