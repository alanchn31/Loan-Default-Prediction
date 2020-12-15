import pandas as pd
import pytest
from app.transformers.get_age import GetAge
from datetime import date

class TestGetAge(object):
    def test_get_age(self, spark_session):
        test_data = spark_session.createDataFrame(
            [(date(1984, 1, 1), date(2018, 8, 3)),
             (date(1977, 12, 9),  date(2018, 9, 26)),
             (date(2000, 6, 1), date(2018, 9, 16))],
             ['DOB', 'DISBURSED_DATE']
        )

        test_config = {
            "age_cols": [{
                 "start": "DOB",
                 "end": "DISBURSED_DATE",
                 "output_col": "BORROWER_AGE"
            }]
        }

        expected_data = pd.DataFrame(
            [(date(1984, 1, 1), date(2018, 8, 3), 34),
             (date(1977, 12, 9),  date(2018, 9, 26), 40),
             (date(2000, 6, 1), date(2018, 9, 16), 18)],
             columns=['DOB', 'DISBURSED_DATE', 'BORROWER_AGE']
        )

        real_data = GetAge(test_config["age_cols"]).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)