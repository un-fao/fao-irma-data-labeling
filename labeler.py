import ee
from datetime import datetime, timedelta
import streamlit as st
import altair as alt
from streamlit_folium import st_folium

from datetime import date
import pandas as pd
import folium
from folium.plugins import Geocoder

from gee_connect import check_gee_connection
from common_functions import *
from constants import valid_s2_dates
from status import display_task_status

import warnings
warnings.filterwarnings('ignore')


print('----------------------\nlabeler\n----------------------')


st.markdown(
    '''<style>
    iframe[title='streamlit_folium.st_folium'] {
            height : calc(100vh - 350px);
    }
    button[kind='secondary'] {
        /*background-color: darkblue;*/
        display: flex;
        flex-wrap: wrap;
    }
    button .e121c1cl0 {
        font-size: 12px;
    }
    button.st-emotion-cache-snstoh.e1obcldf1 {
        min-height: 1.5rem;
    }
    button.st-emotion-cache-b0y9n5.e1obcldf2 {
        min-height: 1.5rem;
    }
    </style>
    ''', unsafe_allow_html=True)


#
#  Common Functions
#

def update_labels():
    if st.session_state.debug:
        print('update labels')
    st.session_state.label_data.loc[int(st.session_state.selected_row), 'label'] = \
        st.session_state.focus_row_changes['edited_rows'][0]['label']
    

def update_manual_labels():
    if st.session_state.debug:
        print('update manual labels')

    check_map_settings()

    st.session_state.updated_label_idx = int(st.session_state.manual_selected_row)
    for row_idx, row in st.session_state.manual_focus_row_changes['edited_rows'].items():
        for key, value in row.items():
            st.session_state.manual_basket_data.loc[st.session_state.updated_label_idx, key] = value

    for row in st.session_state.manual_focus_row_changes['added_rows']:
        st.session_state.manual_basket_data = pd.concat([st.session_state.manual_basket_data, pd.DataFrame([row])])

    for row_idx in st.session_state.manual_focus_row_changes['deleted_rows']:
        st.session_state.manual_basket_data = st.session_state.manual_basket_data.drop(row_idx)
        if st.session_state.manual_selected_row == row_idx:
            st.session_state.manual_selected_row = None
            del st.session_state['updated_label_idx']
        del st.session_state['manual_map']

    st.session_state.manual_selected_row = None
    st.session_state.manual_basket_data.reset_index(drop=True, inplace=True)


def valid_location(loc):
    if st.session_state.debug:
        print('valid location')
    return st.session_state.AOI.geometry().intersects(loc).getInfo()


def check_map_settings():
    if st.session_state.debug:
        print('check map settings')
    if ('render' not in st.session_state) or ('center' not in st.session_state.render):
        return
    if ((st.session_state.manual_label_latitude != st.session_state.render['center']['lat']) and 
        (st.session_state.manual_label_longitude != st.session_state.render['center']['lng'])):
        st.session_state.manual_label_longitude = st.session_state.render['center']['lng']
        st.session_state.manual_label_latitude = st.session_state.render['center']['lat']
    st.session_state.manual_zoom_level = st.session_state.render['zoom']


def add_legend():
    if st.session_state.debug:
        print('add legend')
    legend_html = '''
    <div style="position: fixed; 
        top: 0px; left: 0px; width: 420px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white; opacity: 0.85;">
        <div style="display: flex; justify-content: center; font-family: sans-serif; ">
            <div style="margin: 5px 5px 5px 20px; width: 48%; text-align: center;">
                Rainfall(mm) prior 15 days<br>
                <div style="width: 220px; display: flex; font-size: 10px; margin: 0 auto;">
                    <span>0mm &nbsp;</span>
                    <div style="width: 150px; height: 10px; 
                        background: linear-gradient(to right, #8B4513, #CD853F, #F5DEB3, #87CEEB, #4682B4, #000080);">
                    </div>
                    <span>&nbsp; 30mm</span>
                </div>
            </div>
            <div style="margin: 5px 5px 5px 20px; width: 48%; text-align: center;">
                NDVI<br>
                <div style="width: 200px; display: flex; font-size: 10px; margin: 0 auto;">
                    <span>0 &nbsp;</span>
                    <div style="width: 150px; height: 10px; 
                        background: linear-gradient(to right, #d73027, #f46d43, #fdae61, #fee08b, #ffffbf, #d9ef8b, #a6d96a, #66bd63, #1a9850, #006837);">
                    </div>
                    <span>&nbsp; 1.0</span>
                </div>
            </div>
        </div>
    </div>
    '''
    st.components.v1.html(legend_html, height=50)


def display_label_class_selector():
    if st.session_state.debug:
        print('display label class selector')
    st.radio('Select Label Class and Click on Map to Add to Basket', ['Irrigated', 'Rainfed'], 
                                index=0, horizontal=True, key='manual_label_class')


def get_nearest_s2_date(target_date):
    if st.session_state.debug:
        print('get nearest s2 date')
    start_date = (target_date - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (target_date + timedelta(days=3)).strftime('%Y-%m-%d')
    s2_img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(start_date, end_date)   
              .filterBounds(st.session_state.AOI.geometry())
              .first())
    s2_date = s2_img.get('system:time_start').getInfo()
    return datetime.fromtimestamp(s2_date/1000).strftime('%Y-%m-%d')


def select_manual_img_date():
    if st.session_state.debug:
        print('select manual img date')
    if 'manual_selected_date' in st.session_state:
        if str(st.session_state.manual_img_date) == str(st.session_state.manual_selected_date):
            return
        if st.session_state.manual_selected_date is None:
            st.session_state.manual_img_date = None
        else:
            st.session_state.manual_img_date = get_nearest_s2_date(st.session_state.manual_selected_date)
        for key in ['manual_map']:
            if key in st.session_state:
                del st.session_state[key]


def clear_manual_basket():
    if st.session_state.debug:
        print('clear manual basket')
    st.session_state.manual_selected_row = None
    for key in ['manual_basket_data', 'manual_map']:
        if key in st.session_state:
            del st.session_state[key]


def new_random_date():
    if st.session_state.debug:
        print('new random date')
    if 'manual_map' in st.session_state:
        del st.session_state['manual_map']
    manual_img_date_index = random.randint(0, len(valid_s2_dates)-1)
    st.session_state.manual_img_date = valid_s2_dates[manual_img_date_index]


def set_manual_selected_row():
    if st.session_state.debug:
        print('set manual selected row')

    if len(st.session_state.manual_basket_df.selection.rows) > 0:
        sel_row = st.session_state.manual_basket_df.selection.rows[0]
        st.session_state.manual_selected_row = sel_row
        st.session_state.manual_label_latitude = st.session_state.manual_basket_data.loc[sel_row, 'latitude']
        st.session_state.manual_label_longitude = st.session_state.manual_basket_data.loc[sel_row, 'longitude']
        prior_date = st.session_state.get('manual_img_date', None)
        st.session_state.manual_img_date = st.session_state.manual_basket_data.loc[sel_row, 'date']
        if prior_date is not None:
            if st.session_state.manual_img_date != prior_date:
                del st.session_state['manual_map']
    else:
        st.session_state.manual_selected_row = None


def click_on_map():
    if st.session_state.debug:
        print('click on map')

    # both of these conditions are needed to check for a new label.  either on its own is not enough
    if (('center' in st.session_state.map_obj) and 
        (st.session_state.render['last_clicked'] is not None)):
        if st.session_state.debug:
            print('click on map: new label')
        x = st.session_state.render['last_clicked']
        if not valid_location(ee.Geometry.Point([x['lng'], x['lat']])):
            st.toast('Select a location within the target region')
        else:
            st.session_state.new_label = True
            try:
                # get image date for the tile in the selected pixel
                img_date = ee.Date(st.session_state.manual_ndvi.select('system:time_start').reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=ee.Geometry.Point([x['lng'], x['lat']]),
                    scale=10
                ).get('system:time_start')).format('YYYY-MM-dd').getInfo()
            except Exception as e:
                st.toast(f'Error getting image date.  This may be due to the selected location not having an true-colorimage tile.  If so, try a different location or date.', icon="🚨")
                time.sleep(3)
                return

            new_label_row = pd.DataFrame({'latitude': [x['lat']], 'longitude': [x['lng']], 'label': [st.session_state.manual_label_class], 'date': [img_date]})
            st.session_state.manual_basket_data = pd.concat([st.session_state.manual_basket_data, new_label_row])
            st.session_state.manual_basket_data.reset_index(drop=True, inplace=True)

            check_map_settings()


def add_label_marker_to_map(label, lat, lng, idx, retain_position=False):
    color = 'red' if label == 'Rainfed' else 'blue' if label == 'Irrigated' else 'gray'
    icon = 'cloud-rain' if label == 'Rainfed' else 'faucet' if label == 'Irrigated' else 'question'
    folium.Marker(location=[lat, lng], popup=f"{str(idx)}:{label}",
                icon=folium.Icon(icon=icon, prefix='fa', color=color)).add_to(st.session_state.manual_map)
    if retain_position:
        st.session_state.manual_map.location = [st.session_state.manual_label_latitude, st.session_state.manual_label_longitude]
        st.session_state.manual_map.options['zoom'] = st.session_state.manual_zoom_level


@st.fragment
def display_everything():
    st.session_state.manual_basket_data = pd.DataFrame(columns=['label', 'longitude', 'latitude', 'date']) if 'manual_basket_data' not in st.session_state else st.session_state.manual_basket_data

    manual_img_date_window = 4
    manual_img_date_start = (datetime.strptime(st.session_state.manual_img_date, '%Y-%m-%d') - timedelta(days=manual_img_date_window)).strftime('%Y-%m-%d')
    manual_img_date_end = (datetime.strptime(st.session_state.manual_img_date, '%Y-%m-%d') + timedelta(days=manual_img_date_window)).strftime('%Y-%m-%d')

    manual_data_panel, manual_map_panel = st.columns([2, 4], border=True)
    
    with manual_data_panel:
        if st.session_state.debug:
            print('manual data panel')

        col1, col2 = st.columns(2)
        col1.write('Select a date')
        with col2:
            if st.session_state.debug:
                print('manual date input')
            manual_date_input = st.date_input(
                "Select a date",
                value=datetime.strptime(st.session_state.manual_img_date, '%Y-%m-%d'),
                min_value=datetime(2018, 1, 1),
                max_value=datetime(2025, 1, 1),
                key='manual_selected_date',
                on_change=select_manual_img_date,
                label_visibility='collapsed'
            )

        st.write(f'Range shown: :green[{manual_img_date_start} - {manual_img_date_end}]')

        manual_col1, manual_col2, manual_col3 = st.columns(3)
        disable_buttons = len(st.session_state.manual_basket_data) < 1
        handle_page_unload(not disable_buttons)
        manual_col1.button('Clear', key=None, help='Warning: unsaved changes are lost!', on_click=clear_manual_basket, icon=':material/delete:', disabled=disable_buttons)
        manual_col2.button('New Date', key=None, help='Generate a new random date', on_click=new_random_date)#, icon=':material/refresh:')
        manual_col3.button('Save Basket', key=None, help=None, on_click=save_manual_labels, type='primary', disabled=disable_buttons)

        st.write(f'Basket (select line to edit label): :green[{len(st.session_state.manual_basket_data)} rows]')
        st.dataframe(
            st.session_state.manual_basket_data.rename_axis('REF'),
            height=int(0.4*800),
            selection_mode="single-row",
            hide_index=False,
            on_select=set_manual_selected_row,
            column_order=['label', 'date', 'longitude', 'latitude'],
            column_config={
                #"date": None,
            },
            key='manual_basket_df',
            width=1000,
        )

        st.session_state.manual_selected_row = None if 'manual_selected_row' not in st.session_state else st.session_state.manual_selected_row
        
        st.text('Label to edit / Add dates for same location')
        edit_data = pd.DataFrame([st.session_state.manual_basket_data.iloc[st.session_state.manual_selected_row]]) if st.session_state.manual_selected_row is not None else pd.DataFrame(columns=['label', 'longitude', 'latitude', 'date'])
        edit_data['date'] = edit_data['date'].apply(lambda x: pd.to_datetime(x).date())
        default_longitude = edit_data['longitude'].values[0] if len(edit_data['longitude'].values) > 0 else None
        default_latitude = edit_data['latitude'].values[0] if len(edit_data['latitude'].values) > 0 else None
        st.session_state.manual_focus_row_data = st.data_editor(
            edit_data.rename_axis('REF'),
            num_rows='dynamic' if len(edit_data) > 0 else 'fixed',
            column_order=['label', 'date', 'longitude', 'latitude'],
            column_config={
                "REF": st.column_config.Column(required=False),
                "label": st.column_config.SelectboxColumn(
                    "label",
                    help="Assessed use of irrigation",
                    width="narrow",
                    options=[
                        "Rainfed",
                        "Irrigated",
                        "Unknown",
                    ],
                    required=True,
                ),
                "date": st.column_config.DateColumn(
                    "date",
                    min_value=date(2018, 1, 1),
                    max_value=date(2025, 1, 1),
                    format="YYYY-MM-DD",
                    step=1,
                    required=True,
                ),
                "longitude": st.column_config.NumberColumn(
                    default=default_longitude,
                ),
                "latitude": st.column_config.NumberColumn(
                    default=default_latitude,
                ),
            },
            key='manual_focus_row_changes',
            disabled=('REF', 'longitude', 'latitude'),
            hide_index=False,
            width=1000,
            #on_change=update_manual_labels,
        )

        def cancel_label_changes():
            st.session_state.manual_selected_row = None
        
        if len(edit_data) > 0:
            edit_col1, edit_col2, edit_col3 = st.columns(3)
            with edit_col2:
                st.button('Cancel', on_click=cancel_label_changes)
            with edit_col3:
                st.button('Save Edits', on_click=update_manual_labels, type='primary')


    with manual_map_panel:
        if st.session_state.debug:
            print('manual map panel')
    
        display_label_class_selector()

        with st.container():
            if 'manual_label_longitude' not in st.session_state:
                if st.session_state.debug:
                    print('manual label longitude not in session state')

                aoi_center = st.session_state.AOI.geometry().centroid().coordinates().getInfo()
                st.session_state.manual_label_longitude = aoi_center[0]
                st.session_state.manual_label_latitude = aoi_center[1]
                st.session_state.manual_zoom_level = 8

            if (('manual_map' not in st.session_state)
                #or ('manual_map_center_changed' in st.session_state)
                ):
                if st.session_state.debug:
                    print('rebuild manual map')

                if 'manual_map_center_changed' in st.session_state:
                    del st.session_state['manual_map_center_changed']
                manual_map = folium.Map(location=[st.session_state.manual_label_latitude, st.session_state.manual_label_longitude], 
                                        zoom_start=st.session_state.manual_zoom_level, 
                                        attr="Manual Mapping")
                folium.TileLayer('OpenStreetMap').add_to(manual_map)
                folium.TileLayer('Esri.WorldImagery', name='Recent Hi-Res Image', show=False).add_to(manual_map)
                
                add_aoi_districts(manual_map)

                st.session_state.manual_ndvi = add_manual_rgb_image_to_map(manual_map, manual_img_date_start, manual_img_date_end)
                add_manual_landcover_image_to_map(manual_map)
                add_manual_ndvi_image_to_map(manual_map, st.session_state.manual_ndvi)
                add_manual_precip_image_to_map(manual_map, st.session_state.manual_img_date)

                folium.LayerControl().add_to(manual_map)

                Geocoder(collapsed=True, position='topleft', placeholder='Search for a location').add_to(manual_map)

                st.session_state.manual_map = manual_map

                for idx, row in st.session_state.manual_basket_data.iterrows():
                    add_label_marker_to_map(row['label'], row['latitude'], row['longitude'], idx, False)

            if 'new_label' in st.session_state:
                if st.session_state.debug:
                    print('new label')

                for idx, row in st.session_state.manual_basket_data.iterrows():
                    add_label_marker_to_map(row['label'], row['latitude'], row['longitude'], idx, True)
                del st.session_state['new_label']
            
            elif 'updated_label_idx' in st.session_state:
                if st.session_state.debug:
                    print('updated label idx')
                    
                idx = st.session_state.updated_label_idx
                row = st.session_state.manual_basket_data.loc[idx]
                add_label_marker_to_map(row['label'], row['latitude'], row['longitude'], idx, True)
                del st.session_state['updated_label_idx']

            # it is critical to use the key, the on_change function, and the map_obj variable 
            # to correctly handle map clicks and adding new labels
            st.session_state.map_obj = st_folium(
                st.session_state.manual_map,
                key='render',
                on_change=click_on_map,
                returned_objects=['last_clicked', 'zoom', 'center'],
                width='100%',
                height=500,
            )
            add_legend()


def main():
    if st.session_state.debug:
        print('main')

    check_gee_connection()

    if (('manual_img_date' not in st.session_state) or 
        (st.session_state.manual_img_date is None)):
        new_random_date()
    
    display_everything()
                
    display_task_status()
    st.write('')  # simple spacer


if 'region' in st.session_state:
    main()