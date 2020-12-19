import pyspark.sql.functions as fn
from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from typing import List


class ExtractTimePeriodMths(Transformer):
    """
    Extract Period of time from string
    for eg: 2yrs 2mon --> 26
    """

    def __init__(self, tenure_cols: List[str] = None):
        super(ExtractTimePeriodMths, self).__init__()
        self._tenure_cols = tenure_cols

    def _transform(self, df: DataFrame) -> DataFrame:
        if self._tenure_cols == [] or self._tenure_cols is None:
            return df
        for column in self._tenure_cols:
            df = df.withColumn(f"{column}_YRS", fn.regexp_extract(column, "(\d)(yrs)", 1))
            df = df.withColumn(f"{column}_MTHS", fn.regexp_extract(column, "(\d)(mon)", 1))
            df = df.withColumn(column, (fn.col(f"{column}_YRS")*12 + fn.col(f"{column}_MTHS")))
            df = df.drop(*[f"{column}_YRS", f"{column}_MTHS"]) 
        return df