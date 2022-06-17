from datetime import date as dt
from datetime import datetime
import pickle
import os

import pandas as pd
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from prefect.logging import get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")  # Q2 -> 11.637
    return

@task
def get_paths(date):
    """Date is expected to be Y-M-D"""
    
    def calculate_month_year(month, year, months_back):
        month_for_split = month - months_back
        year_for_split = year - 1 if month_for_split <= 0 else year
        return month_for_split, year_for_split
    
    def create_filepath(year, month):
        DATA_PATH = "data/" 
        file = f"fhv_tripdata_{year}-{month:02d}.parquet"
        filepath = os.path.join(DATA_PATH, file)
        return filepath
    
    MONTHS_BACK_FOR_TRAINING = 2
    MONTHS_BACK_FOR_VALIDATION = 1
    
    date_dt = datetime.strptime(date, '%Y-%m-%d').date() if date else dt.today()
        
    month_train, year_train = calculate_month_year(date_dt.month, date_dt.year, MONTHS_BACK_FOR_TRAINING)    
    month_val, year_val = calculate_month_year(date_dt.month, date_dt.year, MONTHS_BACK_FOR_VALIDATION)
    
    train_file = create_filepath(year_train, month_train)
    val_file = create_filepath(year_val, month_val)
    
    return train_file, val_file
                                               
@flow
def main(date="2021-08-15"):

    categorical = ['PUlocationID', 'DOlocationID']
    
    # Get the train and validation paths
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()  # Q1
    
    # save the trained model and the dict vectorizer
    
    def save_artifact(filepath, artifact):
        # This could be a new task inside the flow
        pickle.dump(artifact, open(filepath, 'wb'))    
    
    base_path = "03-orchestration"
    model_filename = os.path.join(base_path, f"model-{date}.pkl")
    dv_filename = os.path.join(base_path, f"dv-{date}.pkl")
    save_artifact(model_filename, lr)
    save_artifact(dv_filename, dv)  # Q3: 13,000 bytes
    
    # run evaluation
    run_model(df_val_processed, categorical, dv, lr)

DeploymentSpec(
    flow=main,
    name="homework",
    schedule=CronSchedule(cron="0 9 15 * *", timezone="America/New_York"),  # Q4, https://crontab-generator.org/
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
# Q5: 3 upcoming runs
# Q6: prefect work-queue ls
