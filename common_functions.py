#  Common Functions
from branca.element import MacroElement
from datetime import datetime
import ee
import folium
import geemap.foliumap as geemap
from jinja2 import Template
import numpy as np
import pandas as pd
import random
import re
import streamlit as st
import time

from gee_connect import check_gee_connection
check_gee_connection()


label_root_path = f'projects/fao-irma-gee/assets/training_data/labels_v3'
consolidation_path = f'{label_root_path}/consolidated'


def handle_page_unload(modified):
    js_code = f"""
    <script>
        let modified = {str(modified).lower()};
        function handleBeforeUnload(event) {{
            if (modified) {{
                event.preventDefault();
                event.returnValue = "You have unsaved changes. Do you really want to leave?";
            }}
        }}
        window.addEventListener("beforeunload", handleBeforeUnload);
        document.addEventListener("DOMContentLoaded", function() {{
            document.addEventListener("change", function() {{
                modified = true;
            }});
        }});
    </script>
    """

    js_code = f"""
    <script>
        let modified = {str(modified).lower()};
        let saveClicked = false;

        document.addEventListener("DOMContentLoaded", function() {{
            function getSaveButton() {{
                let buttons = parent.document.querySelectorAll("div.stButton button");
                for (let button of buttons) {{
                    let buttonText = button.innerText || button.textContent;  // Get text including nested elements
                    if (buttonText.includes("Save")) {{
                        return button;  // Found the button
                    }}
                }}
                return null;  // Not found yet
            }}

            // Function to attach event listener once button appears
            function attachSaveListener() {{
                let saveButton = getSaveButton();
                if (saveButton) {{
                    saveButton.addEventListener("click", function() {{
                        saveClicked = true;
                        modified = false;  // Reset modification flag on save
                    }});
                }} else {{
                    setTimeout(attachSaveListener, 500);  // Retry after 500ms if button not found
                }}
            }}

            attachSaveListener(); // Start looking for the button

            // Warn user if they try to leave with unsaved changes
            window.addEventListener("beforeunload", function(event) {{
                if (modified && !saveClicked) {{
                    event.preventDefault();
                    event.returnValue = "You have unsaved changes. Do you really want to leave?";
                }}
            }});
        }});
    </script>
    """
    
    st.components.v1.html(js_code, height=0)


@st.cache_data
def load_sentinel2_collection(start_date='2018-01-01', end_date='2025-01-01'):
    import ee
    import random
    def maskS2clouds(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask)

    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(start_date, end_date)
              #.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              # .map(maskS2clouds)
              .randomColumn(seed=random.randint(1, 1234567890))
              .sort('random')
              )
    return s2_col

s2_col_full = load_sentinel2_collection()


def add_aoi_districts(map):
    district_style = {'color': '#FF00FF', 'fillOpacity': 0, 'weight': 1}
    folium.GeoJson(
        data=st.session_state.AOI_DISTRICTS,
        name='FAO Level2 Outlines',
        style_function=lambda x: district_style,
        tooltip=folium.GeoJsonTooltip(fields=['ADM2_NAME'], labels=False),
        show=False
    ).add_to(map)


def create_cropland_mask():
    # Option 1
    lgrip = ee.Image('projects/fao-irma-gee/assets/LGRIP30')
    crop_mask = lgrip.select('b1').eq(2).Or(lgrip.select('b1').eq(3))

    # Option 2 - NOT USED
    if False:
        dataset = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')
        land_cover = dataset.select('discrete_classification')
        probability = dataset.select('discrete_classification-proba')
        cropland_mask = land_cover.eq(40)  # cropland value
        high_probability_mask = probability.gte(25)
        crop_mask = cropland_mask.And(high_probability_mask)

    # Option 3 - NOT USED
    if False:
        crop_mask = ee.Image('projects/fao-irma-gee/assets/training_data/masks/Nature_Global_Cropland_1k')

    crop_mask = crop_mask.selfMask()
    st.session_state.cropland_mask = crop_mask if st.session_state.region == 'global' else crop_mask.clip(st.session_state.AOI)


def setup_masks():
    create_cropland_mask()
    get_cropland_locations()

    st.session_state.esa_land_cover = ee.ImageCollection("ESA/WorldCover/v100").mosaic().eq(40).selfMask()
    st.session_state.esa_land_cover = st.session_state.esa_land_cover if st.session_state.region == 'global' else st.session_state.esa_land_cover.clip(
        st.session_state.AOI)


def add_eelayer_to_map(self, image, vis_params={}, name=None, show=True, opacity=0.5, name_key=None):
    map_id_dict = ee.Image(image).getMapId(vis_params)
    url = map_id_dict["tile_fetcher"].url_format
    tile = folium.raster_layers.TileLayer(tiles=url,
                                          attr="Google Earth Engine",
                                          name=name,
                                          overlay=True,
                                          control=True,
                                          show=show,
                                          opacity=opacity,
                                          max_zoom=24, )
    tile.add_to(self)
    return tile


folium.Map.add_eelayer_to_map = add_eelayer_to_map


class CustomControl(MacroElement):
    """Put any HTML on the map as a Leaflet Control.
    Adopted from https://github.com/python-visualization/folium/pull/1662
    Adopted from geemap
    """
    _template = Template(
        """
        {% macro script(this, kwargs) %}
        L.Control.CustomControl = L.Control.extend({
            onAdd: function(map) {
                let div = L.DomUtil.create('div');
                div.innerHTML = `{{ this.html }}`;
                return div;
            },
            onRemove: function(map) {
                // Nothing to do here
            }
        });
        L.control.customControl = function(opts) {
            return new L.Control.CustomControl(opts);
        }
        L.control.customControl(
            { position: "{{ this.position }}" }
        ).addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """
    )

    def __init__(self, html, position="bottomleft"):
        def escape_backticks(text):
            """Escape backticks so text can be used in a JS template."""
            import re
            return re.sub(r"(?<!\\)`", r"\`", text)

        super().__init__()
        self.html = escape_backticks(html)
        self.position = position


def initialize_map(zoom=13):
    lc_vis_params = {'bands': ['Map'], 'palette': ['orange']}

    longitude = st.session_state.label_longitude
    latitude = st.session_state.label_latitude

    m = geemap.Map(location=[latitude, longitude], zoom_start=zoom, plugin_Draw=False)
    
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('Esri.WorldImagery', name='Recent Hi-Res Image', show=False).add_to(m)

    m.add_eelayer_to_map(st.session_state.cropland_mask, {'palette': 'brown'}, 'Cropland', False, 0.5)
    m.add_eelayer_to_map(st.session_state.esa_land_cover, lc_vis_params, 'ESA WorldCover 10m v100', False, 0.5)

    add_aoi_districts(m)

    folium.Marker([latitude, longitude], tooltip="Point of Interest").add_to(m)
    st.session_state['map'] = m


def load_sentinel2_data(longitude, latitude, buffer_km=0.1):
    s2_loc_images = s2_col_full.filterBounds(ee.Geometry.Point([longitude, latitude]).buffer(buffer_km * 1000))
    s2_loc_images = [s2_loc_images.toList(10).get(i) for i in range(10)]
    return s2_loc_images


def clear_labels():
    for key in ['label_data', 'selected_row', 'manual_basket_data', 'manual_map', 
                #'manual_label_longitude', 'manual_img_date', 
                'manual_selected_row']:
        if key in st.session_state:
            del st.session_state[key]



def save_labels():
    run_datetime_str = datetime.now().strftime('%Y-%m-%d')
    run_datetime_ms = datetime.now().strftime('%s')
    region = st.session_state.region.lower().replace(' ', '-')

    def create_ee_feature(row):
        return ee.Feature(
            ee.Geometry.Point([row['longitude'], row['latitude']]),
            {
                'userID': st.session_state.username,
                'region': region,
                'YEAR': pd.to_datetime(row['date']).strftime('%Y'),
                'month': pd.to_datetime(row['date']).strftime('%m'),
                'date': row['date'],
                'POINT_TYPE': 'Unknown' if str(row['label']) == 'nan' else re.sub('[^a-zA-Z]', '', str(row['label'])),
                'labelsaved': run_datetime_ms
            }
        )

    labels = st.session_state.label_data.apply(create_ee_feature, axis=1).tolist()
    fc_labels = ee.FeatureCollection(labels)

    gee_path = f'projects/fao-irma-gee/assets/training_data/labels_v3/{region}/labels'
    task = ee.batch.Export.table.toAsset(
        fc_labels,
        f"Save labels for {st.session_state.username} on {run_datetime_str} at {run_datetime_ms}",
        f"{gee_path}_{run_datetime_str}_{run_datetime_ms}"
    )
    try:
        task.start()
        st.toast('Save submitted! Check Status section below for progress.', icon="✅")
        if 'task_queue' not in st.session_state:
            st.session_state.task_queue = []
        st.session_state.task_queue.append(task)
        clear_labels()
    except Exception as e:
        if st.session_state.debug:
            print('Error saving labels:', str(e))
        st.error(f'ERROR:{str(e)}')

def save_manual_labels():
    st.session_state.label_data = st.session_state.manual_basket_data
    st.session_state.manual_selected_row = None
    save_labels()

def save_reviews():
    run_datetime_str = datetime.now().strftime('%Y-%m-%d')
    run_datetime_ms = datetime.now().strftime('%s')
    region = st.session_state.region.lower().replace(' ', '-')

    def create_ee_feature(row):
        return ee.Feature(
            ee.Geometry.Point([row['longitude'], row['latitude']]),
            {key: row[key] for key in row.keys() if key not in ['longitude', 'latitude']}
        )

    labels = st.session_state.review_data.apply(create_ee_feature, axis=1).tolist()
    fc_labels = ee.FeatureCollection(labels)

    msg = f"Save reviews for {st.session_state.username} on {run_datetime_str} at {run_datetime_ms}"
    task = ee.batch.Export.table.toAsset(
        fc_labels,
        msg,
        f"{consolidation_path}/{region}_{run_datetime_str}_{run_datetime_ms}"
    )
    try:
        task.start()
        st.toast('Save submitted! Check Status tab for progress.', icon="✅")
        if 'task_queue' not in st.session_state:
            st.session_state.task_queue = []
        st.session_state.task_queue.append(task)

    except Exception as e:
        if st.session_state.debug:
            print(e)
        st.toast(str(e), icon="🚨")


def load_preprocessed_label_data(folder=None):
    if folder is None:
        if st.session_state.debug:
            print('Error:  must specify stage')
        return

    data_path = f"{label_root_path}/{folder}"
    label_files = ee.data.listAssets(data_path)
    if len(label_files) < 1:
        return None, None

    region = st.session_state.region.lower().replace(' ', '-')
    label_files = [x['id'] for x in label_files['assets'] if region in x['name']]
    if len(label_files) < 1:
        return None, None

    label_features = [ee.FeatureCollection(x).getInfo() for x in label_files]
    label_data = [item for feature_col in label_features for item in feature_col['features']]

    property_keys = {key for x in label_data for key in x['properties'].keys()}
    labels = pd.DataFrame({
        'longitude': [x['geometry']['coordinates'][0] for x in label_data],
        'latitude': [x['geometry']['coordinates'][1] for x in label_data],
        **{key: [x['properties'].get(key, None) for x in label_data] for key in property_keys}
    })
    return labels, label_files


def get_last_consolidation_data(consolidation_path):
    consolidation_files = ee.data.listAssets(consolidation_path)
    if len(consolidation_files) < 1:
        return None, None

    region = st.session_state.region.lower().replace(' ', '-')
    consolidation_files = [x['id'] for x in consolidation_files['assets'] if region in x['name']]
    if len(consolidation_files) < 1:
        return None, None

    consolidation_files.sort()
    latest_consolidation_id = consolidation_files[-1]

    last_consolidation_features = [ee.FeatureCollection(latest_consolidation_id).getInfo()]
    conso_data = [item for feature_col in last_consolidation_features for item in feature_col['features']]

    property_keys = {key for x in conso_data for key in x['properties'].keys()}
    conso_labels = pd.DataFrame({
        'longitude': [x['geometry']['coordinates'][0] for x in conso_data],
        'latitude': [x['geometry']['coordinates'][1] for x in conso_data],
        **{key: [x['properties'].get(key, None) for x in conso_data] for key in property_keys}
    })
    return conso_labels


def delete_asset(asset_id):
    ee.data.deleteAsset(asset_id)
    # time.sleep(0.5)  # for testing
    return f"Asset '{asset_id}' successfully deleted."


def retrieve_precip_data(metadate, location, window_days=15):
    start_date = ee.Date(metadate).advance(-window_days, 'day')
    end_date = ee.Date(metadate).advance(-1, 'day')
    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    def calc_precip(day_offset, start_):
        start = start_.advance(day_offset, 'days')
        end = start.advance(1, 'days')
        return collection.select('precipitation') \
            .filterDate(start, end) \
            .first() \
            .set('system:time_start', start.millis())

    def extract_precip(image):
        precip_value = image.select('precipitation').reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=location.centroid(),
            scale=ee.Number(image.projection().nominalScale())
        ).get('precipitation')

        precip_value = ee.Number(precip_value)

        return ee.Feature(None, {
            'precip_value': precip_value,
            'date': image.date().format('yyyy-MM-dd'),
            'system:time_start': image.date().millis()
        })

    try:
        number_of_days = end_date.difference(start_date, 'days')
        daily = ee.ImageCollection(
            ee.List.sequence(0, number_of_days).map(lambda x: calc_precip(x, start_date))
        )
        precip_ts = ee.FeatureCollection(daily.map(extract_precip)).sort('date', False)
        precip_data = pd.DataFrame({
            "Date": np.array(precip_ts.aggregate_array('date').getInfo()),
            "Precipitation (mm)": np.array(precip_ts.aggregate_array('precip_value').getInfo())
        })
        return precip_data
    except Exception as e:
        if st.session_state.debug:
            print(f"Error processing precipitation data")
        return pd.DataFrame()



def retrieve_ndvi_data(img_date, location, buffer_km=1):
    start_date = ee.Date(img_date).update(month=1, day=1)
    end_date = start_date.advance(1, 'year')

    def extract_ndvi(image):
        try:
            day = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=location,
                scale=250  # MODIS resolution
            ).get("DayOfYear")

            ndvi = ee.Number(image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=location,
                scale=250  # MODIS resolution
            ).get("NDVI")).multiply(0.0001)
            return ee.Feature(None, {"day": day, "ndvi": ndvi})

        except Exception as e:
            if st.session_state.debug:
                print(f"Error processing image: {e}")
            return ee.Feature(None, {"day": None, "ndvi": None})

    modis_ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filter(ee.Filter.notNull(["system:time_start"]))
        .filterDate(start_date, end_date)
        .filterBounds(location)
    )

    # create dataframe of ndvi timeseries
    ndvi_features = modis_ndvi.map(extract_ndvi).getInfo()["features"]
    dates = [feature["properties"]["day"] for feature in ndvi_features]
    ndvi_values = [feature["properties"]["ndvi"] for feature in ndvi_features]
    filtered_data = [(date, ndvi) for date, ndvi in zip(dates, ndvi_values) if ndvi is not None and ndvi >= 0]
    ndvi_df = pd.DataFrame(filtered_data, columns=["Day", "NDVI"])
    ndvi_df = ndvi_df.sort_values("Day")
    return ndvi_df


def get_cropland_locations(n=250):
    vectorized = st.session_state.cropland_mask.reduceToVectors(
        reducer=ee.Reducer.countEvery(),
        geometryType='centroid',
        geometry=st.session_state.AOI,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
    )
    locations = vectorized.randomColumn(seed=random.randint(1, 3141590112358)).sort('random')
    st.session_state.candidate_locations = locations.limit(n)


def get_random_cropland_location():
    """Returns (longitude, latitude)"""
    idx = random.randint(0, 249)
    return ee.Feature(st.session_state.candidate_locations[idx]).geometry().coordinates().getInfo()


def get_random_locations(n=20):
    locs = []
    i = 0
    c = 0
    tmp = st.session_state.candidate_locations.getInfo()['features']
    for x in tmp:
        if i >= n or c >= len(tmp):
            break

        if x['properties']['label'] == 40 and x['properties']['label'] > 1:
            locs.append(x['geometry']['coordinates'])
            i += 1
        c += 1
    return locs


def get_label_candidates(n_locs=5, n_dates=10):
    def extract_images(loc):
        """gets a list of images for a location for various dates"""
        return ee.ImageCollection(
            s2_col_full.filterBounds(loc.geometry())
                .limit(n_dates)
                .map(lambda x: x.set('custom::geom', loc.geometry().coordinates()))
            )

    s2_imgs = ee.ImageCollection(st.session_state.candidate_locations.limit(n_locs).map(extract_images).flatten())
    st.session_state.rgb_layers = s2_imgs.toList(99)

    def convert_to_dataframe(img):
        img = ee.Image(img)
        data = {}
        data['long-lat'] = img.get('custom::geom')
        data['date'] = img.date().format("YYYY-MM-dd")
        return ee.Dictionary(data)

    tmp = st.session_state.rgb_layers.map(convert_to_dataframe).getInfo()
    for x in tmp:
        x['longitude'] = x['long-lat'][0]
        x['latitude'] = x['long-lat'][1]
        del x['long-lat']
    df_ = pd.DataFrame(tmp)
    df_['label'] = None
    return df_


def get_rgb_image(label_idx=None):
    if label_idx is not None and label_idx >= 0:
        img = ee.Image(st.session_state.rgb_layers.get(label_idx))
        st.session_state.img_date = st.session_state.label_data.iloc[label_idx]['date']
    else:
        img = ee.Image(st.session_state.s2_col.pop())
        st.session_state.rgb_layers.append(img)
        st.session_state.img_date = img.date().format("YYYY-MM-dd").getInfo()
        st.session_state.label_data = pd.concat(
            [st.session_state.label_data, pd.DataFrame({
                'longitude': [st.session_state.label_longitude],
                'latitude': [st.session_state.label_latitude],
                'date': st.session_state.img_date})],
            ignore_index=True
        )
    return img


def add_rgb_image_to_map(_img):
    rgb_sentinel = _img.visualize(bands=["B4", "B3", "B2"], min=0, max=3000)
    st.session_state.map.addLayer(rgb_sentinel, {}, f'Satellite Image for {st.session_state.img_date}')


def add_manual_landcover_image_to_map(map):
    """Adds a landcover image to the map"""
    landcover_vis_params = {
        'palette': ['orange'],
    }

    img = ee.Image('projects/fao-irma-gee/assets/LGRIP30').clip(st.session_state.AOI)
    img = img.select('b1').eq(2).Or(img.select('b1').eq(3)).selfMask()

    map_id_dict = ee.Image(img).getMapId(landcover_vis_params)
    url = map_id_dict['tile_fetcher'].url_format

    tile_layer = folium.raster_layers.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
        name='Cropland Mask',
        overlay=True,
        control=True,
        opacity=0.5,
        show=False,
    )
    tile_layer.add_to(map)

def add_manual_rgb_image_to_map(map, start_date, end_date=None):
    """Adds a RGB image to the map for a given date range

    start_date is a string in the format 'YYYY-MM-DD'
    end_date is optional, if not provided, the next month is used
    """
    def maskclouds(image):
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        qa = image.select('QA60')
        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask)
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    img = (
        collection.filterDate(start_date, end_date)
        .filterBounds(st.session_state.AOI)
        .map(lambda img: img.addBands(img.metadata('system:time_start')))
        .mosaic()
        .select('B4', 'B3', 'B2', 'B8', 'QA60', 'system:time_start')
    ).clip(st.session_state.AOI)

    ndvi = (
        maskclouds(img)
        .normalizedDifference(['B8', 'B4'])
        .rename('NDVI')
        .addBands(img.select('system:time_start'))
    )

    vis_params = {
        'min': 0,
        'max': 2500,
        'bands': ['B4', 'B3', 'B2'],
    }
    map_id_dict = ee.Image(img).getMapId(vis_params)
    url = map_id_dict['tile_fetcher'].url_format
    tile_layer = folium.raster_layers.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
        name='Satellite Image for Date',
        overlay=True,
        control=True,
        opacity=1.0,
        show=True,
    )
    tile_layer.add_to(map)
    return ndvi


def add_manual_ndvi_image_to_map(map, ndvi):
    """Adds a NDVI image to the map for a given date range"""
    # palette ref:  https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi-on-vegetation-natural_colours/
    vis_params = {
        'min': 0.0,
        'max': 1.0,
        'bands': ['NDVI'],
        'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
    }
    ndvi = ndvi.updateMask(ndvi.gt(0))
    try:
        map_id_dict = ee.Image(ndvi).getMapId(vis_params)
        url = map_id_dict['tile_fetcher'].url_format
        tile_layer = folium.raster_layers.TileLayer(
            tiles=url,
            attr="Google Earth Engine",
            name='NDVI',
            overlay=True,
            control=True,
            opacity=0.5,
            show=False,
        )
        tile_layer.add_to(map)
    except Exception as e:
        if st.session_state.debug:
            print(f'No NDVI data available for this date: {e}')

def add_manual_precip_image_to_map(map, img_date):
    """Adds a precipitation image to the map showing 15-day accumulated precipitation"""
    start_date = ee.Date(img_date).advance(-15, 'days')
    end_date = ee.Date(img_date)

    precip = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .filterDate(start_date, end_date)
            .filterBounds(st.session_state.AOI)
            .sum()
            .clip(st.session_state.AOI)
    )

    vis_params = {
        'min': 0,
        'max': 100,
        'palette': ['#8B4513', '#CD853F', '#F5DEB3', '#87CEEB', '#4682B4', '#000080']
    }

    map_id_dict = precip.getMapId(vis_params)
    url = map_id_dict['tile_fetcher'].url_format
    
    tile_layer = folium.raster_layers.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
        name='15-day Precipitation',
        overlay=True,
        control=True,
        opacity=0.5,
        show=False
    )
    tile_layer.add_to(map)

