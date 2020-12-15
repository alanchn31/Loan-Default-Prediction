import pandas as pd
import pytest
from app.transformers.extract_time_period_mths import ExtractTimePeriodMths

class TestExtractTimePeriodMths(object):
    def test_extract_time_period_mths(self, spark_session):
        test_data = spark_session.createDataFrame(
            [("1yrs 7mon", "0yrs 0mon"),
            ("2yrs 2mon", "1yrs 0mon")],
             ["AVG_LOAN_TENURE", "CREDIT_HIST_LEN"]
        )

        test_config = {
            "tenure_cols": ["CREDIT_HIST_LEN", "AVG_LOAN_TENURE"]
        }

        expected_data = pd.DataFrame(
            [(19, 0),
             (26, 12)],
             columns=["AVG_LOAN_TENURE", "CREDIT_HIST_LEN"]
        )

        real_data = ExtractTimePeriodMths(test_config['tenure_cols']).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)