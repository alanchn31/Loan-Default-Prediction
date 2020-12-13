import pandas as pd
from app.jobs import model_train

class TestModelTrainJob:
    def test_remove_duplicates(self, spark_session):
        test_data = spark_session.createDataFrame(
            [(1, 25, "Male"),
             (1, 25, "Male"),
             (2, 25, "Male"),
             (3, 34, "Female"),
             (1, 38, "Female")],
             ['ID', 'Age', 'Gender']
        )

        expected_data = spark_session.createDataFrame(
            [(25, "Male"),
             (34, "Female"),
             (38, "Female")],
             ['Age', 'Gender']
        ).toPandas()

        real_data = model_train._remove_duplicates(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data[['Age', 'Gender']],
                                      expected_data,
                                      check_dtype=False)
        assert real_data.ID.nunique() == len(real_data.ID)