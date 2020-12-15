import pyspark.sql.functions as fn
from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from typing import List


class DropColumns(Transformer):
    """Converts date string column to date column"""

    def __init__(self, cols: List[str] = None):
        super(DropColumns, self).__init__()
        self._cols = cols

    def _transform(self, df: DataFrame) -> DataFrame:
        if self._cols == [] or self._cols is None:
            return df
        df = df.drop(*self._cols)
        return df