import pandas as pd
import pyspark.sql.types as typ
import pytest
from app.jobs import model_train
from datetime import date

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
        

    def test_check_numerical_dtype(self, spark_session):
        test_data = spark_session.createDataFrame(
            [("Male", 10)],
            ["Gender", "Age"]
        )
        with pytest.raises(Exception, match="Non-numeric value found in column"):
            model_train._check_numerical_dtype(test_data, "Gender")
    

    def test_remove_outliers(self, spark_session):
        test_data = spark_session.createDataFrame(
            [(1, -100),
             (2, 20),
             (3, 30),
             (4, 40),
             (5, 100)],
             ['ID', 'Age']
        )

        test_config = {
            "winsorize_cols": ['Age']
        }

        expected_data = spark_session.createDataFrame(
            [(1, -10),
             (2, 20),
             (3, 30),
             (4, 40),
             (5, 70)],
             ['ID', 'Age']
        ).toPandas()

        real_data = model_train._impute_outliers(test_data, test_config).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)
    

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

        real_data = model_train._convert_str_to_date(test_data, test_config).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)