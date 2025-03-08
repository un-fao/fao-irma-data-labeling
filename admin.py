import altair as alt
from datetime import datetime
import ee
from gee_connect import check_gee_connection
from common_functions import *
import pandas as pd
import streamlit as st
import time


print('----------------------\nadmin\n----------------------')



#
#  Common Functions
#

def count_consolidated_labels():
    consolidation_files = ee.data.listAssets(consolidation_path)
    if len(consolidation_files) < 1:
        return 0

    consolidation_files = [x['id'] for x in consolidation_files['assets'] if
                           st.session_state.region.lower().replace(' ', '-') in x['name']]
    if len(consolidation_files) < 1:
        return 0

    consolidation_files.sort()
    latest_consolidation_id = consolidation_files[-1]

    last_consolidation_features = [ee.FeatureCollection(latest_consolidation_id).getInfo()]
    conso_data = [item for feature_col in last_consolidation_features for item in feature_col['features']]
    return len(conso_data)


@st.fragment(run_every=3)
def display_status_container():
    if 'task' not in st.session_state or not st.session_state.task.active():
        st.session_state.status_container = st.status('No tasks running', state='complete', expanded=False)
    else:
        st.session_state.status_container


def consolidate_labels(new_labels, new_asset_ids):
    prior_labels = get_last_consolidation_data(consolidation_path)
    all_labels = pd.concat([prior_labels, new_labels])

    if 'modified_by' in all_labels.columns:
        all_labels['modified_by'] = all_labels.modified_by.fillna('')
    else:
        all_labels['modified_by'] = ''

    all_labels['reviewed'] = all_labels['modified_by']

    def create_ee_feature(row):
        return ee.Feature(
            ee.Geometry.Point([row['longitude'], row['latitude']]),
            {key: row[key] for key in row.keys() if key not in ['longitude', 'latitude']}
        )

    labels = all_labels.apply(create_ee_feature, axis=1).tolist()
    fc_labels = ee.FeatureCollection(labels)

    run_datetime_str = datetime.now().strftime('%Y-%m-%d')
    run_datetime_ms = datetime.now().strftime('%s')

    msg = f"Consolidate labels for {st.session_state.region.lower().replace(' ', '-')} on {run_datetime_str} at {run_datetime_ms}"
    print(msg)
    try:
        task = ee.batch.Export.table.toAsset(
            fc_labels,
            msg,
            f"{consolidation_path}/{st.session_state.region.lower().replace(' ', '-')}_{run_datetime_str}_{run_datetime_ms}"
        )
        st.session_state.task = task

        status_bar = st.session_state.status_container
        status_bar.update(label='Consolidating labels', state='running', expanded=True)
        status_bar.write(f"Step 1/2: Consolidation launched")

        task.start()
        while task.active():
            refresh_interval = 15
            time.sleep(refresh_interval)

        status = task.status()
        # status = {'state':'COMPLETED'}  # for testing

        if status['state'] == 'COMPLETED':
            status_bar.write("Step 1/2: Completed!")
        elif status['state'] == 'FAILED':
            status_bar.update(label="Step 1 failed. Contact support.", state='error', expanded=False)
        else:
            status_bar.update(label="Step 1 ended abnormally.  Contact support.", state='error', expanded=False)

        if status['state'] == 'COMPLETED':
            status_bar.write(f"Step 2/2: Cleaning up data")
            for i, asset_id in enumerate(new_asset_ids):
                result = delete_asset(asset_id)
            status_bar.write(f"Removed {len(new_asset_ids)} items")

            status_bar.write("Step 2/2: Cleanup process completed")
            status_bar.update(label='Consolidation completed', state='complete', expanded=False)

            st.rerun()

    except Exception as e:
        print('Error consolidating:', e)
        st.toast(str(e), icon="🚨")


def get_label_stats_data():
    consolidation_files = ee.data.listAssets(consolidation_path)
    if len(consolidation_files) < 1:
        return pd.DataFrame()

    region_names = list(set([x['id'].split('/')[-1].split('_')[0] for x in consolidation_files['assets']]))

    data = pd.DataFrame()
    for region in region_names:
        conso_files = [x['id'] for x in consolidation_files['assets'] if region in x['id']]
        if len(conso_files) > 0:
            conso_files.sort()
            latest_consolidation_id = conso_files[-1]  # most recent

            last_consolidation_features = [ee.FeatureCollection(latest_consolidation_id).getInfo()]
            conso_data = [item for feature_col in last_consolidation_features for item in feature_col['features']]

            property_keys = {key for x in conso_data for key in x['properties'].keys()}
            conso_labels = pd.DataFrame({
                'longitude': [x['geometry']['coordinates'][0] for x in conso_data],
                'latitude': [x['geometry']['coordinates'][1] for x in conso_data],
                **{key: [x['properties'].get(key, None) for x in conso_data] for key in property_keys}
            })
            data = pd.concat([data, conso_labels]) if len(conso_labels) > 0 else data
    return data


def display_stats_charts():
    df_stats = get_label_stats_data()
    label_stats_tab, user_stats_tab = st.tabs(['Label Stats', 'User Stats'])
    with label_stats_tab:
        st.subheader('Showing data for :green[consolidated] labels for ALL regions')
        if df_stats.empty:
            st.caption('No consolidated data available to summarize')
        else:
            st.text('Distribution of labels by class')
            chart = (
                alt.Chart(df_stats[['date', 'region', 'POINT_TYPE']].groupby(['region', 'POINT_TYPE']).count().reset_index())
                .mark_bar()
                .encode(
                    alt.X('POINT_TYPE').axis(title='Class', labelAngle=0),
                    alt.Y('date').axis(title='Count', labelAngle=0),
                    color='region'
                )
            )
            st.altair_chart(chart, use_container_width=True)

            st.text('Distribution of labels by month')
            chart = (
                alt.Chart(df_stats[['date', 'region', 'month']].groupby(['region', 'month']).count().reset_index())
                .mark_bar()
                .encode(
                    alt.X('month').axis(title='Month of Year', labelAngle=0),
                    alt.Y('date').axis(title='Count', labelAngle=0),
                    color='region'
                )
            )
            st.altair_chart(chart, use_container_width=True)

            st.text('Distribution of labels by year')
            chart = (
                alt.Chart(df_stats[['date', 'region', 'YEAR']].groupby(['region', 'YEAR']).count().reset_index())
                .mark_bar()
                .encode(
                    alt.X('YEAR').axis(title='Year', labelAngle=0),
                    alt.Y('date').axis(title='Count', labelAngle=0),
                    color='region'
                )
            )
            st.altair_chart(chart, use_container_width=True)

        with user_stats_tab:
            st.subheader('Showing data for :green[consolidated] labels for ALL regions')
            if df_stats.empty:
                st.caption('No consolidated data available to summarize')
            else:
                df_stats['labelsaved'] = pd.to_datetime(df_stats['labelsaved'], unit='s')  # date label saved

                st.text('Running sum count of labels per user per week')
                df = df_stats.groupby([pd.Grouper(key='labelsaved', freq='W-MON'), 'userID']).size().rename('count').reset_index()
                df['week'] = df['labelsaved'].dt.isocalendar().week
                df['year'] = df['labelsaved'].dt.isocalendar().year
                df['cumulative_count'] = df.groupby(['userID', 'year'])['count'].cumsum()
                pivot_df = df.pivot_table(
                    index=['year', 'week'],
                    columns='userID',
                    values='cumulative_count',
                    fill_value=0
                )
                pivot_df.index = pivot_df.index.map(lambda x: f"{x[0]} - W {x[1]}")
                st.line_chart(pivot_df)

                st.text('Count of labels by labeler')
                st.bar_chart(df_stats.groupby(['userID']).count()['POINT_TYPE'])


def main():
    check_gee_connection()

    label_mgt_tab, stats_tab = st.tabs(['Manage Labels', 'View Stats'])
    with label_mgt_tab:
        unconsolidated_labels, unconsolidated_assets = load_preprocessed_label_data(folder='reviewed')
        num_labels = 0 if unconsolidated_labels is None else len(unconsolidated_labels)

        col1, col2, col3 = st.columns(3)
        col1.metric(label='Number of :blue[unconsolidated] labels', value=num_labels)
        col2.metric(label='Number of :green[consolidated] labels', value=count_consolidated_labels())

        if num_labels > 0:
            confirm_consolidate = st.button('Consolidate labels')
            if confirm_consolidate:
                consolidate_labels(unconsolidated_labels, unconsolidated_assets)

            st.caption('or')

            col_confirm, col_bt, _, _ = st.columns(4)
            confirm_delete = col_confirm.checkbox(':red[Delete] unconsolidated labels without saving?')
            if confirm_delete and col_bt.button('Confirm deletion?', type='primary'):
                st.info(f"Deleting unconsolidated labels")
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, asset_id in enumerate(unconsolidated_assets):
                    status_text.text(f"Deleting {asset_id} ({i + 1}/{len(unconsolidated_assets)})...")
                    result = delete_asset(asset_id)
                    st.write(result)
                    progress_bar.progress((i + 1) / len(unconsolidated_assets))
                st.success("Deletion process completed!")

        display_status_container()


    with stats_tab:
        if st.button('Show Charts', key="show_stats_charts"):
            display_stats_charts()

if 'region' in st.session_state:
    main()
