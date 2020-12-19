import pyspark.sql.functions as fn
from pyspark.sql import DataFrame
from pyspark.ml import Transformer


class ReplaceStrRegex(Transformer):
    """Replace string in a String column using regex"""

    def __init__(self, str_replace_cols: dict = None):
        super(ReplaceStrRegex, self).__init__()
        self._str_replace_cols = str_replace_cols

    def _transform(self, df: DataFrame) -> DataFrame:
        if self._str_replace_cols == {} or self._str_replace_cols is None:
            return df
        for column, val in self._str_replace_cols.items():
            pattern = val["pattern"]
            replacement = val["replacement"]
            df = df.withColumn(column, fn.regexp_replace(column, pattern, replacement))
        return df