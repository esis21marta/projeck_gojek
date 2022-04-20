"""
This is a boilerplate pipeline
generated using Kedro 0.18.0
"""
################# Import Library ######################
import logging
import numpy as np
import pandas as pd
import os
import sys
import pandas as pd
import geopy.distance
from haversine import haversine
import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple, List
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score 
from kedro.extras.datasets import json,pickle

DATE_FMT = "%Y-%m-%d %H:%M:%S.%f %Z"
################# make dataset ######################
def clean_booking_df(df:pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    df = df.drop_duplicates(subset=parameters["unique_columns"])
    return df.loc[:,parameters["unique_columns"]]
def clean_participant_df(df:pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df
def merge_dataset(df1: pd.DataFrame, df2: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    df = pd.merge(df1, df2, 
    on=parameters["on_column_1"], 
    how=parameters["how_column_1"])
    return df
def create_target(df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    df[parameters["target_column"]] = df["participant_status"].apply(lambda x: int(x == "ACCEPTED"))
    return df
################# Feature Engineering ######################
def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df
def driver_distance_to_pickup_with_geopy(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance_geopy"] = df.apply(
        lambda r: geopy.distance.geodesic(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ).m,
        axis=1,
    )
    return df
def driver_historical_completed_bookings_train(df: pd.DataFrame,parameters: Dict[str, Any]) -> pd.DataFrame:
    try:
        df1 = df[df[parameters["target_column"]]==1]
        df1 = df1.groupby('driver_id',as_index=False)['order_id'].count().rename(columns={'order_id':'num_of_completed_driver'})
        df = pd.merge(df, df1, on="driver_id", how="left")
        df['num_of_completed_driver'] = df['num_of_completed_driver'].fillna(0)
        return df
    except:
        raise NotImplementedError(
        f"Show us your feature engineering skills! Suppose that drivers with a good track record are more likely to accept bookings. "
        f"Implement a feature that describes the number of bookings that each driver has completed."
    )    
def iso_to_datetime(iso_str: str, date_format: str = DATE_FMT) -> datetime:
    return datetime.strptime(iso_str, date_format)
def hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    return iso_to_datetime(iso_str, date_format).hour
def robust_hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    try:
        return hour_of_iso_date(iso_str, date_format)
    except:
        return hour_of_iso_date(iso_str, "%Y-%m-%d %H:%M:%S %Z")
def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df
def driver_historical_completed_bookings_test(df: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    try:
        df1 = df1[['driver_id','num_of_completed_driver']].drop_duplicates()
        df = pd.merge(df, df1, on="driver_id", how="left")
        df['num_of_completed_driver'] = df['num_of_completed_driver'].fillna(0)
        return df
    except:
        raise NotImplementedError(
        f"Show us your feature engineering skills! Suppose that drivers with a good track record are more likely to accept bookings. "
        f"Implement a feature that describes the number of bookings that each driver has completed.")
################# Build Modelling ######################
class SklearnClassifier:
    def __init__(self, estimator: BaseEstimator, features: List[str], target: str,):
        self.clf = estimator
        self.features = features
        self.target = target
    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)
        return self
    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]
    def evaluate(self, df_test: pd.DataFrame):
        try:
            metric_score = roc_auc_score(df_test[self.target],self.predict(df_test))
            return metric_score
        except:
            raise NotImplementedError(
            f"You're almost there! Identify an appropriate evaluation metric for your model and implement it here."
            f"The expected output is a dictionary of the following schema: {{metric_name: metric_score}}")

def model_building(df:pd.DataFrame, parameters: Dict[str, Any]):
    #prepare dataset
    df_train, df_test = train_test_split(df, test_size=parameters["test_size"])
    #tuning hyperparameter model
    random_forest = RandomForestClassifier(random_state=123)
    hyperparam = {
        'min_samples_split': [2,4,5,8,10,12,15],
        'max_features': [0.01,0.25,0.5,0.75],
        'n_estimators': [10,20,30,40,60],
        }
    random_rf = RandomizedSearchCV(random_forest, 
                                             param_distributions = hyperparam,
                                             cv = 5, 
                                             n_iter = 10, 
                                             scoring = 'accuracy', 
                                             n_jobs=-1, 
                                             random_state = 123)
    best_rf = SklearnClassifier(random_rf, parameters["features"], parameters["target_column"])
    best_rf.train(df_train)
    # Best Selected parameter for random forest Model
    randomforest = RandomForestClassifier(random_state=123, 
    n_jobs = -1,
    min_samples_split = best_rf.clf.best_params_.get('min_samples_split'),
    max_features = best_rf.clf.best_params_.get('max_features'),
    n_estimators = best_rf.clf.best_params_.get('n_estimators'))
    model = SklearnClassifier(randomforest, parameters["features"], parameters["target_column"])
    model.train(df_train)
    #evaluation metrics with AUC Score
    metric_score_train = model.evaluate(df_train)
    metric_score_test = model.evaluate(df_test)
    #create dictionary
    metrics = {'AUC_SCORE_TRAIN': metric_score_train,'AUC_SCORE_TEST': metric_score_test}
    #store the results
    res = []
    res.append(model)
    res.append(metrics)
    return res
################# predicted ######################
def choose_best_driver(df: pd.DataFrame, model: pickle.PickleDataSet) -> pd.DataFrame:
    df["score"] = model.predict(df)
    df['driver_id'] = df['driver_id'].astype('str')
    df['order_id'] = df['order_id'].astype('str')
    df = df.groupby("order_id").agg({"driver_id": list, "score": list}).reset_index()
    df["best_driver"] = df.apply(lambda r: r["driver_id"][np.argmax(r["score"])], axis=1)
    df["order_id"] = df["order_id"].astype('str')
    df["best_driver"] = df["best_driver"].astype('str')
    df = df[['order_id','driver_id','best_driver','score']]
    df = df.drop(["driver_id","score"], axis=1)
    df = df.rename(columns={"best_driver": "driver_id"})
    return df
    
# def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
#     accuracy = (y_pred == y_test).sum() / len(y_test)
#     logger = logging.getLogger(__name__)
#     logger.info("Model has accuracy of %.3f on test data.", accuracy)
