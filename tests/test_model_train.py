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


    def test_impute_missing_values(self, spark_session):
        test_data = spark_session.createDataFrame(
            [(1, 20, 100, None),
             (2, 30, 200, "Male"),
             (3, 40, None, "Male"),
             (4, None, 300, "Female")],
             ['ID', 'Age', 'Amount', 'Gender']
        )

        test_config = {
            "impute_cols": {
                'Age': 'median',
                'Amount': 'mean',
                'Gender': 'missing'
            }
        }

        expected_data = spark_session.createDataFrame(
            [(1, 20.0, 100.0, "missing"),
             (2, 30.0, 200.0, "Male"),
             (3, 40.0, 200.0, "Male"),
             (4, 30.0, 300.0, "Female")],
             ['ID', 'Age', 'Amount', 'Gender']
        ).toPandas()

        real_data = model_train._impute_missing_values(test_data, test_config).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)