import pandas as pd
import pytest
from app.transformers.remove_duplicates import RemoveDuplicates

class TestRemoveDuplicates(object):
    def test_remove_duplicates(self, spark_session):
        test_data = spark_session.createDataFrame(
            [(1, 25, "Male"),
             (1, 25, "Male"),
             (2, 25, "Male"),
             (3, 34, "Female"),
             (1, 38, "Female")],
             ['ID', 'Age', 'Gender']
        )

        test_config = {
            "id_col": "ID"
        }

        expected_data = spark_session.createDataFrame(
            [(25, "Male"),
             (34, "Female"),
             (38, "Female")],
             ['Age', 'Gender']
        ).toPandas()

        real_data = RemoveDuplicates(test_config['id_col']).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data[['Age', 'Gender']],
                                      expected_data,
                                      check_dtype=False)
        assert real_data.ID.nunique() == len(real_data.ID)