import pandas as pd
import numpy as np
from meteostat import Point, Stations, Hourly
import datetime

from great_tables import GT, md, style, loc

import altair as alt
from vega_datasets import data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def get_data():
    stations = Stations()
    locs=[ (40.4406, -79.9959),
    (33.9425, -118.4081),
    (25.7959, -80.2870),
    (39.8617, -104.6731),
    (41.9742, -87.9073),
    (47.4502, -122.3088)]

    def _get_nearby_stations(latitude, longitude, n=6):  
        stations = Stations()
        
        stations = stations.nearby(latitude, longitude)
        stations = stations.fetch(n)    

        #rank stations based on distance to main station 
        stations["rank"] = stations["distance"].rank(method="first").astype(int)

        #print(stations.head())


        return stations.reset_index()
    def _get_data(stations, start = datetime.datetime(2024, 1, 1), end = datetime.datetime(2024, 12, 31)):
    
        station_ids = stations['id'].tolist()
        data = Hourly(station_ids, start, end).fetch().reset_index()

        # attach main station ID to all rows BEFORE pivot
        main_station = stations.iloc[0]["id"]
        data["main_station"] = main_station

        # Merge with station coords/rank
        data = data.merge(
            stations[["id", "longitude", "latitude", "rank"]],
            left_on="station",
            right_on="id",
            how="left"
        )

        # compute u/v
        data["u"] = data["wspd"] * np.sin(np.deg2rad(data["wdir"] + 180))
        data["v"] = data["wspd"] * np.cos(np.deg2rad(data["wdir"] + 180))
        data = data[["time","rank","main_station","u","v","longitude","latitude"]]

        # average duplicate rows
        data = data.groupby(["time","rank","main_station"], as_index=False).mean()

        # pivot, but KEEP main_station!
        data = data.pivot(index=["time","main_station"], columns="rank", values=["u","v","longitude","latitude"])

        # flatten
        data.columns = [f"{var}_{rank}" for var, rank in data.columns]
        data = data.dropna().reset_index()

        # rename for consistency
        data = data.rename(columns={"main_station": "station"})

        return data

    
    all_stations=pd.DataFrame()
    data=pd.DataFrame()
    for lat, lon in locs:
        stations_loc = _get_nearby_stations(lat, lon, n=6)
        data_loc = _get_data(stations_loc)
        all_stations = pd.concat([all_stations, stations_loc]).drop_duplicates()
        data = pd.concat([data, data_loc]).drop_duplicates()
        
    stations = all_stations.reset_index()
    #print(data.columns)

    return data, stations


def train_model(data):

    y = data[["u_1", "v_1"]]
    station = data[["station"]]

    # Drop non-numeric features
    X = data.drop(columns=["u_1", "v_1", "station", "time"])

    # Remove rows with NaNs
    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    station = station[mask]

    # Split
    X_train, X_test, y_train, y_test, station_train, station_test = train_test_split(
        X, y, station, test_size=0.2, shuffle=True, random_state=42
    )

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    return station_test, y_test.to_numpy(), y_pred


def visualize_predictions(station_test, y_test, y_pred, stations):
    alt.data_transformers.disable_max_rows()

    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Build dataframe
    wind_df = pd.DataFrame({
        'u_obs': y_test[:, 0],
        'v_obs': y_test[:, 1],
        'u_pred': y_pred[:, 0],
        'v_pred': y_pred[:, 1],
        'station': station_test.values.ravel()
    })

    wind_df = wind_df.merge(
        stations[['id', 'longitude', 'latitude']],
        left_on='station',
        right_on='id',
        how='left'
    )
    
    scale = .5
    wind_df["end_lon"] = wind_df["longitude"] + wind_df["u_obs"] * scale
    wind_df["end_lat"] = wind_df["latitude"] + wind_df["v_obs"] * scale
    wind_df["end_lon_pred"] = wind_df["longitude"] + wind_df["u_pred"] * scale
    wind_df["end_lat_pred"] = wind_df["latitude"] + wind_df["v_pred"] * scale
    wind_df.sort_values(by="station", inplace=True)

    wind_df = wind_df.reset_index(drop=True).reset_index().rename(columns={"index": "arrow_id"})
    

    # Slider parameter
    selector = alt.param(
        name="selected_arrow",
        bind=alt.binding_range(min=0, max=len(wind_df)-1, step=1, name="Arrow:"),
        value=0
    )

    # US map
    us_states = alt.topo_feature(data.us_10m.url, "states")

    base = (
        alt.Chart(us_states)
        .mark_geoshape(fill="#f0f0f0", stroke="black")
        .project("albersUsa")
        .interactive()
    )

    # Stations
    points = (
        alt.Chart(stations)
        .mark_circle(size=200, opacity=0.8)
        .encode(
            longitude="longitude:Q",
            latitude="latitude:Q",
            color="rank:N",
            tooltip=["name:N", "id:N", "longitude:Q", "latitude:Q",
                     "elevation:Q", "rank:N", "distance:Q"]
        )
    )

    # Observed arrow
    arrows_obs = (
        alt.Chart(wind_df)
        .mark_line()
        .encode(
            longitude='longitude:Q',
            latitude='latitude:Q',
            longitude2='end_lon:Q',
            latitude2='end_lat:Q',
            color=alt.value("blue"),
            strokeWidth=alt.value(4)
        )
        .transform_filter("datum.arrow_id == selected_arrow")
        
    )

    # Predicted arrow
    arrows_pred = (
        alt.Chart(wind_df)
        .mark_line()
        .encode(
            longitude='longitude:Q',
            latitude='latitude:Q',
            longitude2='end_lon_pred:Q',
            latitude2='end_lat_pred:Q',
            color=alt.value("red"),
            strokeWidth=alt.value(4)
        )
        .transform_filter("datum.arrow_id == selected_arrow")
    )

    # Layer all, attach params ONCE
    map_chart = (
        (base + points + arrows_obs + arrows_pred)
        .add_params(selector)
        .properties(
            width=600,
            height=400,
            title="Observed vs Predicted Wind Vector"
        )
    )

    return map_chart


def make_wind_model_charts():
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import datetime, timedelta
    from meteostat import Point, Hourly
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.ensemble import HistGradientBoostingRegressor
    np.random.seed(42)

    study_locations = {
        'Chicago': (41.8781, -87.6298),
        'Denver': (39.7392, -104.9903),
        'Miami': (25.7617, -80.1918),
        'Phoenix': (33.4484, -112.0740),
        'Seattle': (47.6062, -122.3321)
    }

    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    def collect_weather_data(locations, start_date, end_date):
        all_data = []
        exclude_vars = ['tsun', 'snow', 'wpgt', 'prcp']
        
        for city_name, (lat, lon) in locations.items():
            try:
                point = Point(lat, lon)
                data = Hourly(point, start_date, end_date).fetch()
                
                try:
                    from meteostat import Stations
                    stations = Stations()
                    stations = stations.nearby(lat, lon)
                    station_data = stations.fetch(1)
                    if not station_data.empty:
                        elevation = station_data.iloc[0]['elevation']
                    else:
                        elevation = None
                except:
                    elevation = None
                
                for col in exclude_vars:
                    if col in data.columns:
                        data = data.drop(columns=[col])
                
                if 'coco' in data.columns:
                    def group_condition_codes(code):
                        if pd.isna(code):
                            return 'Unknown'
                        code = int(code)
                        if code in [1, 2]:
                            return 'Clear'
                        elif code in [3, 4, 5, 6]:
                            return 'Cloudy/Foggy'
                        elif code in [7, 8, 9, 17]:
                            return 'Rain'
                        elif code in [10, 11, 12, 13, 14, 15, 16, 19, 21]:
                            return 'Cold Precip'
                        elif code in [18, 20, 22, 23, 24, 25, 26, 27]:
                            return 'Extreme'
                        else:
                            return 'Other'
                    
                    data['condition_group'] = data['coco'].apply(group_condition_codes)
                
                data['city'] = city_name
                data['latitude'] = lat
                data['longitude'] = lon
                data['elevation'] = elevation
                data['datetime'] = data.index
                data = data.reset_index()
                
                all_data.append(data)
                    
            except Exception as e:
                pass
                
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            empty_cols = []
            for col in combined_data.columns:
                if combined_data[col].isna().all():
                    empty_cols.append(col)
            
            if empty_cols:
                combined_data = combined_data.drop(columns=empty_cols)
            
            return combined_data
        else:
            return pd.DataFrame()

    weather_data = collect_weather_data(study_locations, start_date, end_date)

    def prepare_wind_direction_features(df):
        df_clean = df.copy()
        
        df_clean['time'] = pd.to_datetime(df_clean['time'])
        df_clean = df_clean.sort_values(['city', 'time']).reset_index(drop=True)
        
        df_clean['hour'] = df_clean['time'].dt.hour
        df_clean['month'] = df_clean['time'].dt.month
        
        # Create circular encodings for hour and month
        df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
        df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
        
        df_clean['month_sin'] = np.sin(2 * np.pi * (df_clean['month'] - 1) / 12)
        df_clean['month_cos'] = np.cos(2 * np.pi * (df_clean['month'] - 1) / 12)
        
        numerical_features = ['temp', 'dwpt', 'rhum', 'pres', 'wspd', 
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                            'latitude', 'longitude', 'elevation']
        categorical_features = ['condition_group', 'city']
        
        available_numerical = [col for col in numerical_features if col in df_clean.columns]
        available_categorical = [col for col in categorical_features if col in df_clean.columns]
        
        for col in available_numerical:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        for col in available_categorical:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
        
        return df_clean, available_numerical, available_categorical

    weather_clean, numerical_cols, categorical_cols = prepare_wind_direction_features(weather_data)

    def prepare_wind_direction_model(df):
        """
        Prepare data for circular wind direction prediction
        """
        
        model_data = df[df['wdir'].notna()].copy()
        
        numerical_features = ['temp', 'dwpt', 'rhum', 'pres', 'wspd', 
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                            'latitude', 'longitude', 'elevation']
        numerical_available = [col for col in numerical_features if col in model_data.columns]
        
        categorical_features = ['condition_group', 'city']
        categorical_available = [col for col in categorical_features if col in model_data.columns]
        
        # Prepare features
        X_numerical = model_data[numerical_available].copy()
        X_categorical = model_data[categorical_available].copy()
        
        # Fill missing values
        X_numerical.fillna(X_numerical.median(), inplace=True)
        X_categorical.fillna(X_categorical.mode().iloc[0], inplace=True)
        
        # Convert wind direction to circular components
        wind_radians = np.radians(model_data['wdir'])
        y_sin = np.sin(wind_radians)
        y_cos = np.cos(wind_radians)
        
        return X_numerical, X_categorical, y_sin, y_cos, numerical_available, categorical_available

    X_numerical, X_categorical, y_sin, y_cos, numerical_cols, categorical_cols = prepare_wind_direction_model(weather_clean)

    if len(categorical_cols) > 0:
        onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_categorical_encoded = onehot_encoder.fit_transform(X_categorical)

        feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
        X_categorical_encoded = pd.DataFrame(X_categorical_encoded, 
                                        columns=feature_names, 
                                        index=X_categorical.index)
        
        X = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    else:
        X = X_numerical.copy()

    # Split data for both sine and cosine components
    X_train, X_test, y_sin_train, y_sin_test = train_test_split(X, y_sin, test_size=0.2, random_state=42)
    _, _, y_cos_train, y_cos_test = train_test_split(X, y_cos, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression models for sine and cosine
    linear_sin = LinearRegression()
    linear_cos = LinearRegression()
    linear_sin.fit(X_train_scaled, y_sin_train)
    linear_cos.fit(X_train_scaled, y_cos_train)

    y_sin_pred_linear = linear_sin.predict(X_test_scaled)
    y_cos_pred_linear = linear_cos.predict(X_test_scaled)

    # Train HistGradientBoosting models for sine and cosine
    hist_sin = HistGradientBoostingRegressor(random_state=42)
    hist_cos = HistGradientBoostingRegressor(random_state=42)
    hist_sin.fit(X_train, y_sin_train)
    hist_cos.fit(X_train, y_cos_train)

    y_sin_pred_hist = hist_sin.predict(X_test)
    y_cos_pred_hist = hist_cos.predict(X_test)

    # Convert back to degrees
    def circular_to_degrees(sin_pred, cos_pred):
        return np.degrees(np.arctan2(sin_pred, cos_pred)) % 360

    y_pred_linear = circular_to_degrees(y_sin_pred_linear, y_cos_pred_linear)
    y_pred_hist = circular_to_degrees(y_sin_pred_hist, y_cos_pred_hist)

    # Convert test data back to degrees for comparison
    y_test_degrees = np.degrees(np.arctan2(y_sin_test, y_cos_test)) % 360

    # Calculate circular R² scores
    def circular_r2(actual_degrees, predicted_degrees):
        # Convert to circular errors
        errors = np.abs(actual_degrees - predicted_degrees)
        circular_errors = np.minimum(errors, 360 - errors)
        
        # Calculate total sum of squares (circular)
        mean_direction = np.degrees(np.arctan2(np.mean(np.sin(np.radians(actual_degrees))), 
                                            np.mean(np.cos(np.radians(actual_degrees))))) % 360
        mean_errors = np.abs(actual_degrees - mean_direction)
        mean_circular_errors = np.minimum(mean_errors, 360 - mean_errors)
        
        ss_tot = np.sum(mean_circular_errors**2)
        ss_res = np.sum(circular_errors**2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    linear_r2 = circular_r2(y_test_degrees, y_pred_linear)
    hist_r2 = circular_r2(y_test_degrees, y_pred_hist)

    # Select best model
    if hist_r2 > linear_r2:
        best_model_name = "HistGradientBoosting"
        best_predictions = y_pred_hist
        best_r2 = hist_r2
    else:
        best_model_name = "Linear Regression"
        best_predictions = y_pred_linear
        best_r2 = linear_r2

    # Sample data for visualization (Altair has 5000 row limit)
    sample_size = min(2000, len(y_test_degrees))
    sample_indices = np.random.choice(len(y_test_degrees), sample_size, replace=False)

    diagonal_line = pd.DataFrame({'x': [0, 360], 'y': [0, 360]})

    # Linear Regression plot
    linear_data = pd.DataFrame({
        'Actual': y_test_degrees.iloc[sample_indices],
        'Predicted': y_pred_linear[sample_indices]
    })

    linear_scatter = alt.Chart(linear_data).mark_circle(size=30, opacity=0.6).encode(
        x=alt.X('Actual:Q', title='Actual Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        y=alt.Y('Predicted:Q', title='Predicted Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        color=alt.value('blue')
    )

    linear_line = alt.Chart(diagonal_line).mark_line(color='red', strokeWidth=3).encode(
        x='x:Q',
        y='y:Q'
    )

    linear_chart = (linear_scatter + linear_line).properties(
        width=400,
        height=400,
        title=f'Linear Regression - R² = {linear_r2:.3f}'
    )

    # HistGradientBoosting plot
    hist_data = pd.DataFrame({
        'Actual': y_test_degrees.iloc[sample_indices],
        'Predicted': y_pred_hist[sample_indices]
    })

    hist_scatter = alt.Chart(hist_data).mark_circle(size=30, opacity=0.6).encode(
        x=alt.X('Actual:Q', title='Actual Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        y=alt.Y('Predicted:Q', title='Predicted Wind Direction (degrees)', scale=alt.Scale(domain=[0, 360], nice=False)),
        color=alt.value('green')
    )

    hist_line = alt.Chart(diagonal_line).mark_line(color='red', strokeWidth=3).encode(
        x='x:Q',
        y='y:Q'
    )

    hist_chart = (hist_scatter + hist_line).properties(
        width=400,
        height=400,
        title=f'HistGradientBoosting - R² = {hist_r2:.3f}'
    )

    return alt.hconcat(linear_chart, hist_chart)