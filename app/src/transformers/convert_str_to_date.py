import pyspark.sql.functions as fn
from pyspark.sql import DataFrame
from pyspark.ml import Transformer


class ConvertStrToDate(Transformer):
    """Converts date string column to date column"""

    def __init__(self, date_str_cols: dict = None):
        super(ConvertStrToDate, self).__init__()
        self._date_str_cols = date_str_cols

    def _transform(self, df: DataFrame) -> DataFrame:
        if self._date_str_cols == {} or self._date_str_cols is None:
            return df
        # Converts "1/1/99" --> "01/01/1999", then convert to date object
        for column, year_prefix in self._date_str_cols.items():
            df = df.withColumn(column, fn.to_date(column, "d/M/yy"))
            df = df.withColumn(column, fn.when(fn.year(column) > (int(year_prefix) + 1)*100, 
                                            fn.add_months(column, -12*100)) \
                                        .when(fn.year(column) == 1900, 
                                            fn.add_months(column, 12*100)) \
                                        .otherwise(fn.col(column)))
        return df