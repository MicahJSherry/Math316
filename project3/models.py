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
    stations = stations.nearby(40.4406, -79.9959)

    stations = stations.fetch(6)

    data = Hourly(stations, datetime.datetime(2024, 1, 1), datetime.datetime(2024, 12, 31))
    data = data.fetch()
    data = data.reset_index()[["station","time", "wspd","wdir"]]


    data = data.pivot(index="time", columns="station", values=["wspd", "wdir"])

    # flatten MultiIndex columns: wspd_XXXX, wdir_XXXX
    data.columns = [f"{var}_{station}" for var, station in data.columns]
    data = data.dropna()


    X = data[['wspd_F2UX6', 'wspd_KAFJ0', 'wspd_KBTP0', 'wspd_KBVI0',
            'wspd_KPJC0',  'wdir_F2UX6', 'wdir_KAFJ0', 'wdir_KBTP0',
            'wdir_KBVI0', 'wdir_KPJC0']]

    y = data[["wspd_72520", "wdir_72520"]]

    stations = stations.reset_index()
    return X, y, stations



def train_model(X,y):
    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,      # 20% test
        shuffle=True,
        random_state=42
    )

    # --------------------------
    # Fit linear regression model
    # --------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --------------------------
    # Predict
    # --------------------------
    y_pred = model.predict(X_test)

    # --------------------------
    # Metrics
    # --------------------------
    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    y_test = y_test.to_numpy()

    return y_test, y_pred

def visualize_predictions(y_test, y_pred, stations):
    # Convert to NumPy array if not already
    y_test = np.array(y_test, dtype=float)  # ensures it's numeric
    y_pred = np.array(y_pred, dtype=float)

    # Now compute observed components
    u_obs = y_test[:, 0] * np.sin(np.deg2rad(y_test[:, 1]+180))
    v_obs = y_test[:, 0] * np.cos(np.deg2rad(y_test[:, 1]+180))

    # Predicted components
    u_pred = y_pred[:, 0] * np.sin(np.deg2rad(y_pred[:, 1]+180))
    v_pred = y_pred[:, 0] * np.cos(np.deg2rad(y_pred[:, 1]+180))


    wind_df = pd.DataFrame({
        'u_obs': u_obs,
        'v_obs': v_obs,
        'u_pred': u_pred,
        'v_pred': v_pred
    })
    # Get scalar longitude/latitude
    lon = stations.loc[stations["id"]=="72520", "longitude"].values[0]
    lat = stations.loc[stations["id"]=="72520", "latitude"].values[0]

    # Repeat for all rows in wind_df
    wind_df["longitude"] = lon
    wind_df["latitude"] = lat

    # calculate vectors  
    scale = 0.025  
    # Convert to NumPy array if not already
    y_test = np.array(y_test, dtype=float)  # ensures it's numeric
    y_pred = np.array(y_pred, dtype=float)

    # Now compute observed components
    u_obs = y_test[:, 0] * np.sin(np.deg2rad(y_test[:, 1]+180))
    v_obs = y_test[:, 0] * np.cos(np.deg2rad(y_test[:, 1]+180))

    # Predicted components
    u_pred = y_pred[:, 0] * np.sin(np.deg2rad(y_pred[:, 1]+180))
    v_pred = y_pred[:, 0] * np.cos(np.deg2rad(y_pred[:, 1]+180))


    wind_df = pd.DataFrame({
        'u_obs': u_obs,
        'v_obs': v_obs,
        'u_pred': u_pred,
        'v_pred': v_pred
    })
    # Get scalar longitude/latitude
    lon = stations.loc[stations["id"]=="72520", "longitude"].values[0]
    lat = stations.loc[stations["id"]=="72520", "latitude"].values[0]

    # Repeat for all rows in wind_df
    wind_df["longitude"] = lon
    wind_df["latitude"] = lat
    wind_df["end_lon"] = wind_df["longitude"] + wind_df["u_obs"] * scale
    wind_df["end_lat"] = wind_df["latitude"] + wind_df["v_obs"] * scale

    wind_df["end_lon_pred"] = wind_df["longitude"] + wind_df["u_pred"] * scale
    wind_df["end_lat_pred"] = wind_df["latitude"] + wind_df["v_pred"] * scale

    wind_df = wind_df.reset_index().rename(columns={"index": "arrow_id"})


    # --- slider parameter ---
    slider = alt.binding_range(
        min=0,
        max=len(wind_df)-1,
        step=1,
        name="Arrow:"
    )

    selector = alt.param(
        name="selected_arrow",
        bind=slider,
        value=0
    )

    # --- map base ---
    us_states = alt.topo_feature(data.us_10m.url, 'states')
    PA_FIPS = "42"

    base = (
        alt.Chart(us_states)
        .mark_geoshape(fill="#f0f0f0", stroke="black")
        .transform_filter(f"datum.id == '{PA_FIPS}'")
        .project("albersUsa")
    )

    # --- stations ---
    points = (
        alt.Chart(stations)
        .mark_circle(size=200, opacity=0.8)
        .encode(
            longitude="longitude:Q",
            latitude="latitude:Q",
            color="name:N",
            tooltip=["name:N", "id:N"]
        )
    )

    # --- wind arrow (filtered to one row) ---
    arrows = (
        alt.Chart(wind_df)
        .mark_line()
        .encode(
            shape=alt.ShapeValue('arrow'),
            longitude='longitude:Q',
            latitude='latitude:Q',
            longitude2='end_lon:Q',
            latitude2='end_lat:Q',
            color= alt.value("blue"),
            strokeWidth=alt.value(4)
        )
        .add_params(selector)
        .transform_filter("datum.arrow_id == selected_arrow")
    )

    arrows_pred =(
        alt.Chart(wind_df)
        .mark_line()
        .encode(
            shape=alt.ShapeValue('arrow'),
            longitude='longitude:Q',
            latitude='latitude:Q',
            longitude2='end_lon_pred:Q',
            latitude2='end_lat_pred:Q',
            color= alt.value("red"),
            strokeWidth=alt.value(4)
        )
        .add_params()
        .transform_filter("datum.arrow_id == selected_arrow")
    )

    # --- final plot ---
    map = (base + points + arrows + arrows_pred).properties(
        width=600,
        height=400,
        title="Observed vs predicted Wind Vector"
    )

    return map 


