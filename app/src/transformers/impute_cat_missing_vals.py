from pyspark.sql import DataFrame
from pyspark.ml import Transformer


class ImputeCategoricalMissingVals(Transformer):
    """Impute missing values in categorical columns"""

    def __init__(self, cat_cols_dict: dict = None):
        super(ImputeCategoricalMissingVals, self).__init__()
        self._cat_cols_dict = cat_cols_dict

    def _transform(self, df: DataFrame) -> DataFrame:
        for column, value in self._cat_cols_dict.items():
            df = df.fillna({column: value})
        return df