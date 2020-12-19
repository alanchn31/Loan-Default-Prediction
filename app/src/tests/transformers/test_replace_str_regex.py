import pandas as pd
import pytest
from src.transformers.replace_str_regex import ReplaceStrRegex

class TestReplaceStrRegex(object):
    def test_replace_str_regex(self, spark_session):
        test_data = spark_session.createDataFrame(
            [(1, 'Not Scored: No Activity seen on the customer (Inactive)'),
             (2, 'F-Low Risk'),
             (3, 'I-Medium Risk'),
             (4, 'Not Scored: Only a Guarantor'),
             (5, 'Not Scored: No Updates available in last 36 months')
            ],
            ['ID', 'SCORE_CATEGORY']
        )

        test_config = {
            "str_replace_cols": {
                "SCORE_CATEGORY": {
                    "pattern": "Not Scored: (.*)",
                    "replacement": "Not Scored"
                }
            }
        }

        expected_data = pd.DataFrame(
            [(1, 'Not Scored'),
             (2, 'F-Low Risk'),
             (3, 'I-Medium Risk'),
             (4, 'Not Scored'),
             (5, 'Not Scored')],
             columns=['ID', 'SCORE_CATEGORY']
        )

        real_data = ReplaceStrRegex(test_config['str_replace_cols']).transform(test_data).toPandas()

        pd.testing.assert_frame_equal(real_data,
                                      expected_data,
                                      check_dtype=False)