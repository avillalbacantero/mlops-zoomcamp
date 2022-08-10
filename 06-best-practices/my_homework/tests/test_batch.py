from datetime import datetime

import pandas as pd

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    categorical = ["PUlocationID", "DOlocationID"]

    df = pd.DataFrame(data, columns=columns)

    output = prepare_data(df, categorical=categorical)

    expected_output = [
        (-1, -1, dt(1, 2), dt(1, 10), 8.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]

    prepared_columns = [
        "PUlocationID",
        "DOlocationID",
        "pickup_datetime",
        "dropOff_datetime",
        "duration",
    ]

    expected_df = pd.DataFrame(expected_output, columns=prepared_columns)

    output.equals(expected_df)
