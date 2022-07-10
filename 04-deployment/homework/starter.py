import argparse

import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

CATEGORICAL_FEATURES = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna(-1).astype('int').astype('str')
    
    return df

def create_artificial_id(df, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def predict(df):
    dicts = df[CATEGORICAL_FEATURES].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return y_pred

def save_predictions(df):
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == "__main__":
    
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Year to process')
    parser.add_argument('--month', type=int, help='Month to process')
    args = parser.parse_args()
    
    year = args.year
    month = args.month
    
    # Set input and output file
    input_file = f"data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    #input_file = f"../../data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    #output_file = f'output/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f"s3://homework-batch/predictions/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    
    # Read and process the data. We create an aritifial unique ID for
    # each sample
    df = read_data(input_file)
    df = create_artificial_id(df, year, month)
    print(f"Data successfully read!")
    
    # Transform and predict the taxi duration
    print("Predicting batch...")
    preds = predict(df)
    print("Batch predicted")
    
    # Compute the mean of the preds
    mean_predicted_duration = preds.mean()
    print(f"Mean predicted duration: {mean_predicted_duration}")
    
    # Create and output DF, and save it
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = preds
    save_predictions(df_result)
    print("Results successfully written!")