import pandas as pd
import numpy as np
from meteostat import Point, Hourly
import datetime
from great_tables import GT, md, style, loc

import altair as alt
from vega_datasets import data

cities_by_region = {
    'Northeast': {
        'New York, NY': (40.7128, -74.0060),
        'Boston, MA': (42.3601, -71.0589),
        'Philadelphia, PA': (39.9526, -75.1652),
        'Pittsburgh, PA': (40.4406, -79.9959),
        'Buffalo, NY': (42.8864, -78.8784)
    },
    'Southeast': {
        'Miami, FL': (25.7617, -80.1918),
        'Atlanta, GA': (33.7490, -84.3880),
        'Charlotte, NC': (35.2271, -80.8431),
        'Jacksonville, FL': (30.3322, -81.6557),
        'Tampa, FL': (27.9506, -82.4572)
    },
    'Midwest': {
        'Chicago, IL': (41.8781, -87.6298),
        'Detroit, MI': (42.3314, -83.0458),
        'Minneapolis, MN': (44.9778, -93.2650),
        'Cleveland, OH': (41.4993, -81.6944),
        'Milwaukee, WI': (43.0389, -87.9065)
    },
    'West': {
        'Los Angeles, CA': (34.0522, -118.2437),
        'San Francisco, CA': (37.7749, -122.4194),
        'Seattle, WA': (47.6062, -122.3321),
        'Portland, OR': (45.5152, -122.6784),
        'Denver, CO': (39.7392, -104.9903)
    }
}

def wind_stats(city_name, lat, lon, start_year=2020, end_year=2024):
    try:
        point = Point(lat, lon)
        start = datetime.datetime(start_year, 1, 1)
        end = datetime.datetime(end_year, 12, 31)
        
        data = Hourly(point, start, end).fetch()
            
        # Calculate averaged stats
        wind_speed_avg = data['wspd'].mean() if 'wspd' in data.columns else np.nan
        # Convert to radians then back to degrees to avoid circular mean issues
        if 'wdir' in data.columns:
            wind_directions = data['wdir'].dropna()
            if len(wind_directions) > 0:
                directions_rad = np.radians(wind_directions)
                
                sin_mean = np.mean(np.sin(directions_rad))
                cos_mean = np.mean(np.cos(directions_rad))
                
                wind_dir_avg = np.degrees(np.arctan2(sin_mean, cos_mean))
                
                # Ensure result is between 0 and 360
                if wind_dir_avg < 0:
                    wind_dir_avg += 360
        else:
            wind_dir_avg = np.nan
        
        return {
            'city_name': city_name,
            'latitude': lat,
            'longitude': lon,
            'avg_wind_speed_kmh': wind_speed_avg,
            'avg_wind_direction_deg': wind_dir_avg,
        }
        
    except Exception as e:
        return {
            'city_name': city_name,
            'latitude': lat,
            'longitude': lon,
            'avg_wind_speed_kmh': np.nan,
            'avg_wind_direction_deg': np.nan,
        }
    
def create_wind_table():
    def wind_stats_local(city_name, lat, lon, start_year=2020, end_year=2024):
        try:
            point = Point(lat, lon)
            start = datetime.datetime(start_year, 1, 1)
            end = datetime.datetime(end_year, 12, 31)
            data = Hourly(point, start, end).fetch()

            wind_speed_avg = data['wspd'].mean() if 'wspd' in data.columns else np.nan

            if 'wdir' in data.columns:
                wind_directions = data['wdir'].dropna()
                if len(wind_directions) > 0:
                    directions_rad = np.radians(wind_directions)
                    sin_mean = np.mean(np.sin(directions_rad))
                    cos_mean = np.mean(np.cos(directions_rad))
                    wind_dir_avg = np.degrees(np.arctan2(sin_mean, cos_mean))
                    if wind_dir_avg < 0:
                        wind_dir_avg += 360
                else:
                    wind_dir_avg = np.nan
            else:
                wind_dir_avg = np.nan

            return {
                'city_name': city_name,
                'latitude': lat,
                'longitude': lon,
                'avg_wind_speed_kmh': wind_speed_avg,
                'avg_wind_direction_deg': wind_dir_avg,
            }
        except Exception:
            return {
                'city_name': city_name,
                'latitude': lat,
                'longitude': lon,
                'avg_wind_speed_kmh': np.nan,
                'avg_wind_direction_deg': np.nan,
            }

    results = []
    for region, cities in cities_by_region.items():
        for city_name, (lat, lon) in cities.items():
            r = wind_stats_local(city_name, lat, lon)
            r['region'] = region
            results.append(r)

    wind_df_local = pd.DataFrame(results)

    def get_direction_color(direction_deg):
        if pd.isna(direction_deg):
            return "#ffffff"
        direction = float(direction_deg) % 360
        cardinals = {0: (0.2, 0.4, 0.9), 90: (0.9, 0.3, 0.3), 180: (0.9, 0.9, 0.2), 270: (0.3, 0.8, 0.3)}
        cardinal_angles = [0, 90, 180, 270]
        distances = []
        for cardinal in cardinal_angles:
            if cardinal == 0:
                dist = min(direction, 360 - direction)
            else:
                dist = abs(direction - cardinal)
            distances.append(dist)
        min_distance = min(distances)
        closest_cardinal = cardinal_angles[distances.index(min_distance)]
        if min_distance == 45:
            return "#ffffff"
        max_distance = 45
        intensity = 1.0 - (min_distance / max_distance)
        min_intensity = 0.1
        intensity = min_intensity + intensity * (1.0 - min_intensity)
        cardinal_color = cardinals[closest_cardinal]
        white = (1.0, 1.0, 1.0)
        final_color = tuple(intensity * cardinal_color[i] + (1 - intensity) * white[i] for i in range(3))
        r, g, b = [int(c * 255) for c in final_color]
        return f"#{r:02x}{g:02x}{b:02x}"

    # Prepare table data
    table_data_sorted = wind_df_local[[
        'city_name', 'region', 'avg_wind_speed_kmh', 'avg_wind_direction_deg'
    ]].copy()
    table_data_sorted['avg_wind_speed_kmh'] = table_data_sorted['avg_wind_speed_kmh'].round(1)
    table_data_sorted['avg_wind_direction_deg'] = table_data_sorted['avg_wind_direction_deg'].round(0)
    table_data_sorted = table_data_sorted.sort_values(['region', 'avg_wind_speed_kmh'], ascending=[True, False]).reset_index(drop=True)
    table_data_sorted['dir_color'] = table_data_sorted['avg_wind_direction_deg'].apply(get_direction_color)

    min_speed = table_data_sorted['avg_wind_speed_kmh'].min()
    max_speed = table_data_sorted['avg_wind_speed_kmh'].max()

    # Build the table
    table = (
        GT(table_data_sorted.drop(columns=['dir_color']))
        .tab_header(
            title=md("**Regional Wind Analysis by Wind Speed and Direction**"),
            subtitle=md("20 Major Cities on 5-Year Hourly Averages (2020-2024)")
        )
        .cols_label(
            city_name="City",
            avg_wind_speed_kmh="Speed (km/h)",
            avg_wind_direction_deg="Direction (°)",
        )
        .tab_spanner(
            label="Wind Statistics",
            columns=["avg_wind_speed_kmh", "avg_wind_direction_deg"]
        )
        .tab_stub(
            rowname_col="city_name",
            groupname_col="region",
        )
        .data_color(
            columns=['avg_wind_speed_kmh'],
            palette=["#f8f9fa", "#e9ecef", "#dee2e6", "#ced4da", "#adb5bd", "#6c757d"],
            domain=[min_speed, max_speed],
        )
    )

    for idx, row in table_data_sorted.iterrows():
        table = table.tab_style(
            style=style.fill(color=get_direction_color(row['avg_wind_direction_deg'])),
            locations=loc.body(columns=['avg_wind_direction_deg'], rows=[idx])
        )

    table = (
        table
        .cols_align(
            align='center',
            columns=['avg_wind_speed_kmh', 'avg_wind_direction_deg']
        )
        .tab_source_note(
            source_note=md("**Data:** Meteostat | **Speed:** Stronger→Darker | **Direction:** 🔵**North** 🔴**East** 🟡**South** 🟢**West**  Closer to single direction→Darker color")
        )
    )

    globals()['wind_df'] = wind_df_local

    return table



def get_tornado_data():
    tornados_df = pd.read_csv("Tornado_Tracks_1950_2017_1_7964592706304725094.csv")
    tornados_df = tornados_df[tornados_df["Year"]==2024]


    cat_4 =  tornados_df[tornados_df["Magnitude"]==4]
    cat_3 = tornados_df[tornados_df["Magnitude"]==3].sample(n=5,random_state=42)
    cat_2 = tornados_df[tornados_df["Magnitude"]==2].sample(n=5,random_state=42)
    cat_1 = tornados_df[tornados_df["Magnitude"]==1].sample(n=5,random_state=42)
    cat_0 = tornados_df[tornados_df["Magnitude"]==0].sample(n=5,random_state=42)

    case_study= pd.concat([cat_4, cat_3, cat_2, cat_1, cat_0])
    #print(case_study.columns)
    case_study[["State Abbreviation", "Tornado Number", "Year","Month","Day", "Time","Magnitude", "Starting Latitude",	"Starting Longitude",	"Ending Latitude",	"Ending Longitude"]]




    def get_tornado_weekly_wind(tornado_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch hourly wind data (speed & direction) for a week surrounding each tornado event.
        
        Args:
            tornado_df (pd.DataFrame): DataFrame with columns:
                ["State Abbreviation", "Tornado Number", "Year", "Month", "Day", "Time",
                "Magnitude", "Starting Latitude", "Starting Longitude", 
                "Ending Latitude", "Ending Longitude"]
                
        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                ['tornado_number', 'state', 'magnitude', 'lat', 'lon', 
                'time', 'wspd', 'wdir']
        """

        all_data = []

        for idx, row in tornado_df.iterrows():
            try:
                # Parse tornado info
                tornado_num = row["Tornado Number"]
                state = row["State Abbreviation"]
                magnitude = row["Magnitude"]
                lat = row["Starting Latitude"]
                lon = row["Starting Longitude"]
                
                # Parse date and time (safe fallback)
                try:
                    tornado_time = datetime.datetime(
                        int(row["Year"]),
                        int(row["Month"]),
                        int(row["Day"])
                    )
                except Exception:
                    #print(f"⚠️ Skipping tornado {tornado_num}: Invalid date")
                    continue

                # Define ±3 days around event
                start = tornado_time - datetime.timedelta(days=3)
                end = tornado_time + datetime.timedelta(days=3)

                #print(f"Fetching wind for Tornado {tornado_num} ({state}) from {start.date()} to {end.date()}")

                # Create a Meteostat point and fetch hourly data
                point = Point(lat, lon)
                data = Hourly(point, start, end).fetch()

                if not data.empty:
                    data = data.reset_index()
                    data["tornado_number"] = tornado_num
                    data["state"] = state
                    data["magnitude"] = magnitude
                    data["lat"] = lat
                    data["lon"] = lon

                    # Keep only relevant columns
                    data = data[[
                        "tornado_number", "state", "magnitude", "lat", "lon",
                        "time", "wspd", "wdir"
                    ]]
                    all_data.append(data)
                #else:
                    
                    #print(f"⚠️ No wind data found for Tornado {tornado_num} ({state})")

            except Exception as e:
                #print(f"❌ Error fetching Tornado {tornado_num}: {e}")
                continue

        if not all_data:
            #print("No data retrieved for any tornado events.")
            return pd.DataFrame(columns=[
                "tornado_number", "state", "magnitude", "lat", "lon", "time", "wspd", "wdir"
            ])

        result = pd.concat(all_data, ignore_index=True)
        result["day_of_year"] = result["time"].dt.day_of_year

        return result
    tornado = get_tornado_weekly_wind(case_study)
    return tornado



def generate_tornado_map(tornado):
    alt.data_transformers.enable("vegafusion")
    # ['tornado_number', 'state', 'magnitude', 'lat', 'lon', 'time', 'wspd', 'wdir']

    # --------------------------------
    # Tornado Locations (unique per tornado)
    # --------------------------------
    df = (
        tornado[['tornado_number', 'state', 'magnitude', 'lat', 'lon']]
        .drop_duplicates(subset=['tornado_number'])
    )

    # --------------------------------
    # HOURLY WIND DATA
    # --------------------------------
    # No grouping, keep each hourly record
    hourly_wind = tornado[['tornado_number', 'time', 'wspd']].copy()
    hourly_wind = hourly_wind.rename(columns={'wspd': 'wind_speed'})

    # --------------------------------
    # INTERACTIVE SELECTION
    # --------------------------------
    brush = alt.selection_point(fields=['tornado_number'])

    # U.S. map base layer
    us_states = alt.topo_feature(data.us_10m.url, feature='states')

    base_map = alt.Chart(us_states).mark_geoshape(
        fill="#f0f0f0", stroke='black'
    ).project("albersUsa").properties(width=900, height=450)

    # Tornado starting points
    points = alt.Chart(df).mark_circle(size=200, opacity=0.8).encode(
        longitude='lon:Q',
        latitude='lat:Q',
        color=alt.Color('magnitude:N'),
        tooltip=['tornado_number:N', 'state:N', 'magnitude:Q']
    ).add_params(
        brush
    )

    map_layer = base_map + points

    # --------------------------------
    # HOURLY LINE CHART (Filtered by Selection)
    # --------------------------------
    wind_chart = alt.Chart(hourly_wind).mark_line(point=True).encode(
        x=alt.X('time:T', title='Time'),
        y=alt.Y('wind_speed:Q', title='Wind Speed (m/s)'),
        color='tornado_number:N',
        tooltip=['tornado_number:N', 'time:T', 'wind_speed:Q']
    ).transform_filter(
        brush
    ).properties(
        width=900,
        height=300,
        title='Hourly Wind Speed (Filtered by Tornado Selection)'
    )

    # --------------------------------
    # FINAL DASHBOARD
    # --------------------------------
    final_chart = alt.vconcat(map_layer, wind_chart).resolve_scale(color='independent')

    return final_chart


def get_hourly_wind(cities: dict, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Fetch hourly wind data (speed & direction) for multiple cities.
    
    Args:
        cities (dict): Mapping of city names to (lat, lon) tuples.
        start (datetime): Start date for data.
        end (datetime): End date for data.
    
    Returns:
        pd.DataFrame: Combined DataFrame with columns:
                      ['city', 'time', 'wspd', 'wdir']
    """
    all_data = []

    for city, (lat, lon) in cities.items():
        try:
            #print(f"Fetching: {city} ({lat}, {lon})")
            point = Point(lat, lon)
            data = Hourly(point, start, end).fetch()

            if not data.empty:
                data = data.reset_index()
                data['city'] = city
                # keep only relevant columns
                data = data[['city', 'time', 'wspd', 'wdir']]
                all_data.append(data)
            #else:
                #print(f"⚠️ No data for {city}")

        except Exception as e:
            #print(f"❌ Error fetching {city}: {e}")
            continue

    if not all_data:
        #print("No data retrieved for any city.")
        return pd.DataFrame(columns=['city', 'time', 'wspd', 'wdir'])

    return pd.concat(all_data, ignore_index=True)

def generate_cities_map(cities,wind_df):
    alt.data_transformers.enable("vegafusion")
    df = pd.DataFrame([
    {'city': name, 'lat': coords[0], 'lon': coords[1]}
    for name, coords in cities.items()
    ])

    time_range = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 12, 31)]


    daily_avg = (
        wind_df.groupby(['city', 'day_of_year'], as_index=False)['wspd']
        .mean()
        .rename(columns={'wspd': 'wind_speed'})
    )
    #print(daily_avg)
    #print(data)


    brush = alt.selection_point(fields=['city'])

    # World map
    world = alt.topo_feature(data.us_10m.url, feature='states')
    base_map = alt.Chart(world).mark_geoshape(
        fill="#f0f0f0", stroke='black'
    ).project("albersUsa").properties(width=900, height=450)

    # City points with selection
    points = alt.Chart(df).mark_circle(size=200, opacity=0.8).encode(
        longitude='lon:Q',
        latitude='lat:Q',
        color='city:N',
        tooltip=['city:N']
    ).add_params(
        brush
    )

    map_layer = base_map + points

    # Wind line chart filtered by selected cities
    wind_chart = alt.Chart(daily_avg).mark_line().encode(
        x='day_of_year:Q',
        y='wind_speed:Q',
        color='city:N',
        tooltip=['city:N', 'day_of_year:Q', 'wind_speed:Q']
    ).transform_filter(
        brush
    ).properties(
        width=900,
        height=300,
        title='Daily Average Wind Speed by City (Filtered by Map Selection)'
    )

    final_chart = alt.vconcat(map_layer, wind_chart)
    return final_chart
