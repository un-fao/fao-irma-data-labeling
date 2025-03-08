import altair as alt
from collections import Counter
from datetime import datetime, timedelta
import ee
import folium
from folium.plugins import FeatureGroupSubGroup, MarkerCluster
import geopandas as gpd
import pandas as pd
import streamlit as st

from gee_connect import check_gee_connection
from common_functions import *

import warnings
warnings.filterwarnings('ignore')


print('----------------------\nreviewer\n----------------------')


#
#  Common Functions
#

def update_review_labels():
    if st.session_state.debug:
        print('update_review_labels')
        print('Changes made:', st.session_state.review_focus_row_changes)

    if len(st.session_state.review_focus_row_changes['edited_rows']) == 0:
        return
    
    st.session_state.review_data.loc[st.session_state.selected_row, 'modified_by'] = st.session_state.username
    for key, val in st.session_state.review_focus_row_changes['edited_rows'][0].items():
        st.session_state.review_data.loc[st.session_state.selected_row, key] = val
    if 'POINT_TYPE' in st.session_state.review_focus_row_changes['edited_rows'][0].keys():
        st.session_state.review_data.loc[st.session_state.selected_row, 'reviewed'] = True


def load_sentinel2_review_image(longitude, latitude, img_date, buffer_km=0.1):
    if st.session_state.debug:
        print('load_sentinel2_review_image')

    # use larger date range to avoid missing images
    start = ee.Date(img_date)#.advance(-1, 'day')
    end = ee.Date(img_date).advance(1, 'day')
    img = ee.ImageCollection(
        s2_col_full
        .filterDate(start, end)
        .filterBounds(ee.Geometry.Point([longitude, latitude]).buffer(buffer_km * 1000))
    ).mosaic()
    return img


def get_review_image(label_idx, label_lng, label_lat, label_date):
    if st.session_state.debug:
        print('get_review_image:', label_idx)

    st.session_state.review_layers = {} if 'review_layers' not in st.session_state else st.session_state.review_layers
    if label_idx not in st.session_state.review_layers.keys():
        #print('get image', label_idx, label_lng, label_lat, label_date)
        img = ee.Image(load_sentinel2_review_image(label_lng, label_lat, label_date))
        st.session_state.review_layers[label_idx] = img
    else:
        img = ee.Image(st.session_state.review_layers[label_idx])
    return img


def add_jitter(value, jitter_amount=0.001):
    return value + random.uniform(-jitter_amount, jitter_amount)


@st.fragment(run_every=3)
def display_status_container():
    if st.session_state.debug:
        print('display_status_container')

    if 'task' not in st.session_state or not st.session_state.task.active():
        st.session_state.status_container = st.status('No tasks running', state='complete', expanded=False)
    else:
        st.session_state.status_container


def save_review_labels():
    if st.session_state.debug:
        print('save_review_labels')

    reviewed_labels = st.session_state.review_data
    old_asset_ids = st.session_state.review_assets
    region = st.session_state.region.lower().replace(' ', '-')

    if 'modified_by' in reviewed_labels.columns:
        reviewed_labels['modified_by'] = reviewed_labels.modified_by.fillna('')
    else:
        reviewed_labels['modified_by'] = ''

    reviewed_labels['reviewed'] = reviewed_labels['modified_by']

    def create_ee_feature(row):
        return ee.Feature(
            ee.Geometry.Point([row['longitude'], row['latitude']]),
            {key: row[key] for key in row.keys() if key not in ['longitude', 'latitude']}
        )

    labels = reviewed_labels.apply(create_ee_feature, axis=1).tolist()
    fc_labels = ee.FeatureCollection(labels)

    run_datetime_str = datetime.now().strftime('%Y-%m-%d')
    run_datetime_ms = datetime.now().strftime('%s')

    msg = f"Saving reviewed labels for {region} on {run_datetime_str} at {run_datetime_ms}"
    print(msg)
    try:
        task = ee.batch.Export.table.toAsset(
            fc_labels,
            msg,
            f"{label_root_path}/reviewed/{region}_reviewed_{run_datetime_str}_{run_datetime_ms}"
        )
        st.session_state.task = task
        
        status_bar = st.session_state.status_container
        status_bar.update(label='Save running', state='running', expanded=True)
        status_bar.write(f"Step 1/2: Adding Labels")

        task.start()
        while task.active():
            refresh_interval = 5
            time.sleep(refresh_interval)
        
        status = task.status()
        print('done')
        # status = {'state':'COMPLETED'}  # for testing

        if status['state'] == 'COMPLETED':
            status_bar.write("Step 1/2: Completed!")
        elif status['state'] == 'FAILED':
            status_bar.update(label="Step 1 failed. Contact support.", state='error', expanded=False)
        else:
            status_bar.update(label="Step 1 ended abnormally.  Contact support.", state='error', expanded=False)

        if status['state'] == 'COMPLETED':
            status_bar.write(f"Step 2/2: Cleaning up data")

            for i, asset_id in enumerate(old_asset_ids):
                result = delete_asset(asset_id)
            status_bar.write(f"Removed {len(old_asset_ids)} items")

            status_bar.write("Step 2/2: Completed!")
            status_bar.update(label='Save completed', state='complete', expanded=False)

            del st.session_state['review_data']
            del st.session_state['review_assets']


    except Exception as e:
        print('Error adding labels:', e)
        status_bar.update(label='Error adding labels.  Contact support.', state='error', expanded=True)
        status_bar.write(str(e))
        st.toast(str(e), icon="🚨")


def display_overview_map(data):
    if st.session_state.debug:
        print('display_overview_map')

    if data.empty:
        st.info('No data found to display.')
        return

    latitude, longitude = st.session_state.AOI.geometry().centroid().coordinates().getInfo()
    m = geemap.Map(location=[longitude, latitude], zoom_start=8, plugin_Draw=False)
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('Esri.WorldImagery', name='Recent Hi-Res Image', show=False).add_to(m)
    m.add_eelayer_to_map(st.session_state.cropland_mask, {'palette': 'brown'}, 'Cropland', False, 0.5)

    add_aoi_districts(m)

    marker_cluster = MarkerCluster(control=False).add_to(m)
    main_group = folium.FeatureGroup(name="All Points").add_to(m)

    # Create FeatureGroupSubGroups for each POINT_TYPE
    irrigated_group = folium.plugins.FeatureGroupSubGroup(main_group, "Irrigated").add_to(m)
    rainfed_group = folium.plugins.FeatureGroupSubGroup(main_group, "Rainfed").add_to(m)
    unknown_group = folium.plugins.FeatureGroupSubGroup(main_group, "Unknown").add_to(m)

    # Create a dictionary to track counts of POINT_TYPE by cluster location
    cluster_counts = {}

    # Add markers with clustering
    for i, row in data.iterrows():
        # Extract latitude and longitude, adding jitter
        latitude = add_jitter(row['latitude'])
        longitude = add_jitter(row['longitude'])

        # Determine the POINT_TYPE a group
        point_type = row.get('POINT_TYPE', 'unknown').lower()
        if point_type == 'irrigated':
            marker_color = 'red'
            group = irrigated_group
        elif point_type == 'rainfed':
            marker_color = 'blue'
            group = rainfed_group
        else:
            marker_color = 'gray'
            group = unknown_group

        # Dynamically create the tooltip content
        tooltip_content = "<br>".join([f"{col}: {row[col]}" for col in row.index])

        # Update the cluster count dictionary
        # Use rounded lat/lon for approximate cluster locations
        cluster_key = (round(latitude, 2), round(longitude, 2))
        if cluster_key not in cluster_counts:
            cluster_counts[cluster_key] = Counter()
        cluster_counts[cluster_key][point_type] += 1

        # Add marker to the cluster
        folium.Marker(
            [latitude, longitude],
            tooltip=folium.Tooltip(tooltip_content, sticky=True),
            icon=folium.Icon(color=marker_color)
        ).add_to(group)

    # Add callouts to show counts for each cluster
    for (lat, lon), counts in cluster_counts.items():
        # Create a callout text with the counts for each POINT_TYPE in table format
        callout_text = """
        <table style="font-size: 12px; color: black; background: white; padding: 5px; border-collapse: collapse; border: 1px solid black; width: 150px;">
            <tr style="font-weight: bold; background: #f0f0f0;">
                <td>Type</td>
                <td>Count</td>
            </tr>
        """ + "".join([f"""
            <tr>
                <td>{key.capitalize()}</td>
                <td>{value}</td>
            </tr>
        """ for key, value in counts.items()]) + "</table>"

        # Add the callout near the cluster center
        tooltip_content = f"""
                <div style="box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); border-radius: 5px;">
                    {callout_text}
                </div>
            """
        folium.map.Marker(
            [lat, lon],
            tooltip=folium.Tooltip(tooltip_content, sticky=True),
            icon=folium.DivIcon(html=f"""
                    <div><svg>
                        <circle cx="20" cy="15" r="10" stroke="black" stroke-width="0" fill="red" />
                            <!-- Italic "i" -->
                            <text x="20" y="20" font-family="Arial, sans-serif" font-size="14" font-style="italic" text-anchor="middle" fill="white">
                            i
                            </text>
                    </svg></div>""")
        ).add_to(m)

    m.to_streamlit(height=600)


def display_edit_panel():
    data_panel, map_panel = st.columns([2, 4], border=True)

    with data_panel:
        st.write('Labels for Review (select line to review label)')

        if 'review_data' not in st.session_state or st.session_state.review_data.empty:
            st.info('No labels found to review.  Try refreshing page or changing region.')
            return

        data = st.session_state.review_data
        
        with st.expander('Filters', expanded=False, icon=":material/tune:"):
            review_filter = st.pills('Review Status', ['Reviewed', 'Not Reviewed'], selection_mode='single', key='review_filter')
            class_filter = st.pills('Classification(s)', data.POINT_TYPE.unique(), selection_mode='multi', key='class_filter')
            labeler_filter = st.pills('Labeler', data.userID.unique(), selection_mode='single', key='labeler_filter')
            reviewer_filter = st.pills('Reviewer', data.modified_by.unique(), selection_mode='single', key='reviewer_filter')

            if review_filter and len(review_filter) > 0:
                reviewed_mask = True if review_filter == 'Reviewed' else False
                data = data[data.reviewed == reviewed_mask]
            data = data[data.POINT_TYPE.isin(class_filter)] if class_filter and len(class_filter) > 0 else data
            data = data[data.userID.isin(labeler_filter)] if labeler_filter and len(labeler_filter) > 0 else data
            data = data[data.modified_by.isin(reviewer_filter)] if reviewer_filter and len(reviewer_filter) > 0 else data

        st.write(f'Total rows: :green[{len(data)}]')

        labels = st.dataframe(
            data.rename_axis('REF'),
            height=int(0.4*800),
            selection_mode="single-row",
            hide_index=False,
            on_select="rerun",
            column_order=['reviewed', 'POINT_TYPE', 'date', 'userID', 'modified_by', 'longitude', 'latitude'],
            column_config={
                "YEAR": None,
                "month": None,
                "labelsaved": None,
                "region": None,
                "POINT_TYPE": "label",
                "userID": "labeler",
            },
            width=1000,
        )

        btn_disabled = len(data) == 0
        handle_page_unload(not btn_disabled)
        st.button('Save All', key=None, help=None, on_click=save_review_labels, type='primary', disabled=btn_disabled)

    with map_panel:
        st.write('Location Displayed for Review (select classification to edit label)')

        st.session_state.selected_row = 0 if 'selected_row' not in st.session_state else st.session_state.selected_row
        st.session_state.selected_row = labels.selection.rows[0] if len(
            labels.selection.rows) > 0 else st.session_state.selected_row

        idx = st.session_state.selected_row
        st.session_state.focus_row_data = st.data_editor(
            pd.DataFrame([st.session_state.review_data.iloc[idx]]).rename_axis('REF'),
            column_config={
                "POINT_TYPE": st.column_config.SelectboxColumn(
                    "Classification",
                    help="Assessed use of irrigation",
                    width="medium",
                    options=[
                        "Rainfed",
                        "Irrigated",
                        "Unknown",
                    ],
                    required=True,
                ),
                "YEAR": None,
                "month": None,
                "labelsaved": None,
                "region": None,
                "userID": "labeler",
                "modified_by": None,
            },
            key='review_focus_row_changes',
            disabled=('longitude', 'latitude', 'date', 'userID'),
            hide_index=False,
            on_change=update_review_labels,
        )

        lbl_long = st.session_state.review_data.iloc[idx]['longitude']
        lbl_lat = st.session_state.review_data.iloc[idx]['latitude']
        lbl_date = st.session_state.review_data.iloc[idx]['date']

        if (('map' not in st.session_state) or
            ('label_longitude' not in st.session_state) or
            (lbl_long != st.session_state.label_longitude) or
            (lbl_lat != st.session_state.label_latitude) or
            (lbl_date != st.session_state.img_date)):
            st.session_state.label_longitude = lbl_long
            st.session_state.label_latitude = lbl_lat
            st.session_state.img_date = lbl_date
            initialize_map()
            img = get_review_image(idx, lbl_long, lbl_lat, lbl_date)
            add_rgb_image_to_map(img)
            folium.LayerControl().add_to(st.session_state.map)  # DO NOT USE WITH ST_FOLIUM !!!

        map_tab, data_tab = st.tabs(['Map', 'Data'])
        with map_tab:
            st.session_state.map.to_streamlit(height=400)

        with data_tab:
            precip_col, ndvi_col = st.columns(2)
            label_location = ee.Geometry.Point([st.session_state.label_longitude, st.session_state.label_latitude])

            with precip_col:
                precip_data = retrieve_precip_data(st.session_state.img_date, label_location)
                st.text('Precipitation in preceding days')
                st.bar_chart(precip_data, x='Date', y='Precipitation (mm)', height=0.3 * 800)

            with ndvi_col:
                ndvi_data = retrieve_ndvi_data(st.session_state.img_date, label_location)
                ndvi_data['Date'] = ndvi_data.Day.apply(
                    lambda doy: datetime(int(st.session_state.img_date[:4]), 1, 1) + timedelta(days=doy - 1))
                st.text(f'NDVI throughout the year {st.session_state.img_date[:4]}')
                ndvi_chart = (
                    alt.Chart(ndvi_data)
                    .mark_line(color="green")
                    .encode(
                        x=alt.X("Date:T", title="Date"),  # scale=alt.Scale(domain=[0, 365]), title="Date"),
                        y=alt.Y("NDVI:Q", title="NDVI")
                    ).properties(
                        height=0.3 * 800,  # Approximate 30% of viewport height (adjustable)
                        width="container",
                    )
                )
                annotation_layer = (
                    alt.Chart(pd.DataFrame({'x': [pd.to_datetime(st.session_state.img_date)]}))
                    .mark_rule(color='violet', size=2)
                    .encode(x='x')
                )
                combined_chart = ndvi_chart + annotation_layer
                st.altair_chart(combined_chart, use_container_width=True)
                st.caption(':violet[Image date]')


def summarize_district_counts(data):
    if st.session_state.debug:
        print('summarize_district_counts')

    if data.empty:
        return pd.DataFrame()

    # Create GeoDataFrame from input data for spatial operations
    points_gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.longitude, data.latitude)
    )
    
    # Convert AOI districts to GeoDataFrame 
    districts_gdf = gpd.GeoDataFrame.from_features(st.session_state.AOI_DISTRICTS['features'])
    
    # Spatial join points with districts
    joined = gpd.sjoin(points_gdf, districts_gdf, how='left', predicate='within')
    
    # Group and count
    district_counts = []
    grouped = joined.groupby(['ADM2_NAME', 'POINT_TYPE']).size().reset_index(name='count')
    
    # Ensure all district-type combinations exist with 0 counts if needed
    for district in districts_gdf.ADM2_NAME:
        for point_type in data['POINT_TYPE'].unique():
            count = grouped[
                (grouped.ADM2_NAME == district) & 
                (grouped.POINT_TYPE == point_type)
            ]['count'].values
            count = count[0] if len(count) > 0 else 0
            district_counts.append({
                'District': district,
                'POINT_TYPE': point_type, 
                'count': count
            })
    district_counts = pd.DataFrame(district_counts)
    return district_counts


def display_stats_charts(data):
    if data.empty:
        st.caption('No data available to summarize')
        return

    st.subheader('Distribution of labels by class')
    chart = (
        alt.Chart(data[['date', 'POINT_TYPE']].groupby('POINT_TYPE').count().reset_index())
        .mark_bar()
        .encode(
            alt.X('POINT_TYPE').axis(title='Class', labelAngle=0),
            alt.Y('date').axis(title='Count', labelAngle=0),
            color='POINT_TYPE:N'  # Add color encoding
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.text('Distribution of labels by month')
    monthly_data = data.groupby(['month', 'POINT_TYPE']).size().reset_index(name='count')
    chart = (
        alt.Chart(monthly_data)
        .mark_bar()
        .encode(
            alt.X('month:O').axis(title='Month of Year', labelAngle=0),
            alt.Y('count:Q').axis(title='Count', labelAngle=0),
            color='POINT_TYPE:N',
            tooltip=['month', 'POINT_TYPE', 'count']
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.text('Distribution of labels by year')
    yearly_data = data.groupby(['YEAR', 'POINT_TYPE']).size().reset_index(name='count')
    chart = (
        alt.Chart(yearly_data)
        .mark_bar()
        .encode(
            alt.X('YEAR:O').axis(title='Year', labelAngle=0),
            alt.Y('count:Q').axis(title='Count', labelAngle=0),
            color='POINT_TYPE:N',
            tooltip=['YEAR', 'POINT_TYPE', 'count']
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.text('Distribution of labels by state/district')
    district_counts = summarize_district_counts(data)
    if district_counts.empty:
        st.caption('No data available to summarize')
        return
    chart = (
        alt.Chart(district_counts)
        .mark_bar()
        .encode(
            alt.X('District:N', axis=alt.Axis(title='District', labelAngle=45, labelOverlap=False)),
            alt.Y('count:Q').axis(title='Count', labelAngle=0),
            color='POINT_TYPE:N',
            tooltip=['District', 'POINT_TYPE', 'count']
        )
    )
    st.altair_chart(chart, use_container_width=True)


def load_all_labels():
    if 'all_labels' not in st.session_state:
        reviewed_labels, _ = load_preprocessed_label_data(folder='reviewed')
        conso_labels = get_last_consolidation_data(consolidation_path)
        st.session_state.all_labels = pd.concat([reviewed_labels, conso_labels])
    return st.session_state.all_labels


@st.fragment(run_every=15)
def keep_alive():
    if st.session_state.debug:
        print('keep_alive')
    return

def main():
    check_gee_connection()

    setup_masks()

    if 'review_data' not in st.session_state:
        region = st.session_state.region.lower().replace(' ', '-')
        review_data, review_assets = load_preprocessed_label_data(folder=region)
        if review_data is not None:
            st.session_state.review_data = review_data
            st.session_state.review_assets = review_assets
            if 'reviewed' not in st.session_state.review_data.columns:
                st.session_state.review_data['reviewed'] = False
            if 'modified_by' not in st.session_state.review_data.columns:
                st.session_state.review_data['modified_by'] = ''
        st.session_state.rgb_layers = []

    edit_tab, overview_tab_draft, overview_tab_all, stats_tab = st.tabs(['Label Reviewer', 'Map Overview (Draft)', 'Map Overview (All)', 'Summary Metrics (All)'])

    with edit_tab:
        display_edit_panel()
        display_status_container()

    with overview_tab_draft:
        if ('review_data' not in st.session_state) or ('AOI' not in st.session_state):
            st.info('No draft data found to display.')
        elif True: # st.button('Show Map', key="show_map_draft"):
            display_overview_map(st.session_state.review_data)

    with overview_tab_all:
        if True: # st.button('Show Map', key="show_map_all"):
            display_overview_map(load_all_labels())

    with stats_tab:
        if True: # st.button('Show Charts', key="show_charts"):
            display_stats_charts(load_all_labels())

    keep_alive()


if 'region' in st.session_state:
    main()
