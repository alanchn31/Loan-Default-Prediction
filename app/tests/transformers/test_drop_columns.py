import pandas as pd
import pytest
from src.transformers.drop_columns import DropColumns

class TestDropColumns(object):
    def test_drop_cols(self, spark_session):
        test_data = spark_session.createDataFrame(
            [("1/1/84", "3/8/18"),
             ("9/12/77", "26/9/18"),
             ("1/6/00", "16/9/18")],
             ['DOB', 'DISBURSED_DATE']
        )

        test_config = {
            "drop_cols": ['DISBURSED_DATE']
        }

        expected_data = pd.DataFrame(
            [("1/1/84"),
             ("9/12/77"),
             ("1/6/00")],
             columns=['DOB']
        )

        real_data = DropColumns(test_config['drop_cols']).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)