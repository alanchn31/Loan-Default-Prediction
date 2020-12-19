import pyspark.sql.functions as fn
from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from typing import List


class GetAge(Transformer):
    """Get Age in years based on 2 date columns"""

    def __init__(self, age_cols: List[dict] = None):
        super(GetAge, self).__init__()
        self._age_cols = age_cols

    def _transform(self, df: DataFrame) -> DataFrame:
        if self._age_cols == {} or self._age_cols is None:
            return df
        for column_dict in self._age_cols:
            start_col = column_dict["start"]
            end_col = column_dict["end"]
            output_col = column_dict["output_col"]
            df = df.withColumn(output_col, 
                               fn.floor(fn.datediff(end_col, start_col)/365.25))
        return df