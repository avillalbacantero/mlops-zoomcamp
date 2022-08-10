from datetime import datetime

import pandas as pd

from batch import get_input_path, save_data


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_save_data_s3():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]

    df = pd.DataFrame(data, columns=columns)

    input_file = get_input_path(year=2021, month=1)

    save_data(df, input_file)

    assert True
