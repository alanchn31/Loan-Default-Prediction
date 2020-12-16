import pandas as pd
import pytest
from app.transformers.impute_cat_missing_vals import ImputeCategoricalMissingVals
from datetime import date

class TestImputeCategoricalMissingVals(object):
    def test_get_age(self, spark_session):
        test_data = spark_session.createDataFrame(
            [("Bob", "architect"),
             (None, None)],
             ['Name', 'Occupation']
        )

        test_config = {"impute_cat_cols": {
                                            "Name": "missing",
                                            "Occupation": "NA"
                                          }
                      }

        expected_data = pd.DataFrame(
            [("Bob", "architect"),
             ("missing", "NA")],
             columns=['Name', 'Occupation']
        )

        real_data = ImputeCategoricalMissingVals(test_config["impute_cat_cols"]).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)