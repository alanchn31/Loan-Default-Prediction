import pandas as pd
import pytest
from datetime import date
from app.transformers.convert_str_to_date import ConvertStrToDate

class TestConvertStrToDate(object):
    def test_convert_str_to_date(self, spark_session):
        test_data = spark_session.createDataFrame(
            [("1/1/84", "3/8/18"),
             ("9/12/77", "26/9/18"),
             ("1/6/00", "16/9/18")],
             ['DOB', 'DISBURSED_DATE']
        )

        test_config = {
            "date_str_cols": {"DOB": "19",
                              "DISBURSED_DATE": "20"
                             }
        }

        expected_data = pd.DataFrame(
            [(date(1984, 1, 1), date(2018, 8, 3)),
             (date(1977, 12, 9), date(2018, 9, 26)),
             (date(2000, 6, 1), date(2018, 9, 16))],
             columns=['DOB', 'DISBURSED_DATE']
        )

        real_data = ConvertStrToDate(date_str_cols=test_config['date_str_cols']).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)