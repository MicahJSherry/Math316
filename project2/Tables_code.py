import pandas as pd
import numpy as np
from meteostat import Point, Hourly
import datetime
from great_tables import GT, md, style, loc

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