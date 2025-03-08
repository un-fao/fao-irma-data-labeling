"""
Usage:
streamlit run app.py
"""
print('==============================\n\napp\n\n==============================')
import streamlit as st
st.set_page_config(layout="wide")

from auth import authenticate, load_auth_config

import warnings
warnings.filterwarnings('ignore')

st.session_state.debug = False
st.session_state.use_st_secrets = False


st.markdown(
    """
    <style>
            .stAppHeader {
                background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                visibility: visible;  /* Ensure the header is visible */
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
    </style>
    """, unsafe_allow_html=True)


def setup_aoi(region):
    if st.session_state.debug:
        print('setup_aoi')
    import ee
    from gee_connect import check_gee_connection
    check_gee_connection()
    country_borders = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0')
    st.session_state.AOI = country_borders.filter(
        ee.Filter.eq('ADM0_NAME', region.replace('_', ' ').title()))

    fao_districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
    aoi_districts = fao_districts.filter(ee.Filter.eq('ADM0_NAME', st.session_state.region.replace('_', ' ').title()))
    st.session_state.AOI_DISTRICTS = aoi_districts.getInfo()


def get_regions():
    if st.session_state.debug:
        print('get_regions')
    import ee
    from gee_connect import check_gee_connection
    check_gee_connection()

    folders = ee.data.listAssets('projects/fao-irma-gee/assets/training_data/labels_v3')
    folders = [x['id'] for x in folders['assets'] if x['type'] == 'FOLDER' and 'consolidated' not in x['id']]
    regions = [x.split('/')[-1] for x in folders]
    regions = [x.replace('-', ' ').title() for x in regions]
    return regions


def change_region():
    if st.session_state.debug:
        print('change_region')
    for key in ['label_data', 'review_data', 'manual_basket_data', 'manual_img_date', 'manual_label_longitude', 'manual_map', 'review_layers']:
        if key in st.session_state:
            del st.session_state[key]


#
#  User Interface for Application
#
def main():
    if st.session_state.authentication_status:
        if st.session_state.debug:
            print('user logged in')
        with st.sidebar:
            st.markdown("""
            <style>
                section[data-testid="stSidebar"][aria-expanded="true"]{
                    display: true;
                }
            </style>
            """, unsafe_allow_html=True)
            
            auth_config = load_auth_config()
            user_config = auth_config['credentials']['usernames'][st.session_state.username]
            if 'auth_regions' not in user_config or not isinstance(user_config['auth_regions'], list) or len(user_config['auth_regions']) == 0:
                st.error('User has no regions assigned.  Contact admin for access.')
                st.stop()

            regions = user_config['auth_regions']
            regions = get_regions() if user_config['auth_regions'][0] == 'ALL' else regions
            region = st.selectbox(
                "Select a region",
                regions,
                index=0,
                on_change=change_region,
            )
            if 'region' not in st.session_state:
                st.session_state.region = region
            else:
                if st.session_state.region != region:
                    if st.session_state.debug:
                        print('region changed')
                    del st.session_state['region']
                    st.session_state.region = region
                    change_region()
                    st.rerun()


            user_name, user_log = st.columns(2)
            with user_name:
                st.write(st.session_state.name)
            with user_log:
                st.session_state.authenticator.logout('Logout')

            if ('roles' in st.session_state) and (st.session_state.roles is not None) and ('admin' in st.session_state.roles):
                st.session_state.debug = st.checkbox('Debug', value=st.session_state.debug)

        setup_aoi(st.session_state.region)

        if ('roles' in st.session_state) and (st.session_state.roles is not None):
            pages = [st.Page(f'./{role}.py', title=role.capitalize()) for role in st.session_state.roles]
            pages.append(pages.pop(0))
            pages.append(st.Page(f'./guide.py', title='Help'))
            #pages.append(st.Page(f'./poc.py', title='POC'))
            
            pg = st.navigation(pages, position="sidebar", expanded=True)
            pg.run()

    elif (st.session_state.authentication_status is False) or (st.session_state.authentication_status is None):
        with st.sidebar:
            st.markdown("""
            <style>
                section[data-testid="stSidebar"][aria-expanded="true"]{
                    display: none;
                }
            </style>
            """, unsafe_allow_html=True)

        if st.session_state.authentication_status is False: 
            st.error('Username/password is incorrect')



if __name__ == '__main__':
    authenticate('app')
    main()