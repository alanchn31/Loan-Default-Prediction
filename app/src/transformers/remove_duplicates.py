import pyspark.sql.functions as fn
from pyspark.sql import DataFrame
from pyspark.ml import Transformer


class RemoveDuplicates(Transformer):
    """Drops duplicate records"""

    def __init__(self, id_col: str = None):
        super(RemoveDuplicates, self).__init__()
        self._id_col = id_col

    def _transform(self, df: DataFrame) -> DataFrame:
        # Drop exact duplicates
        df = df.dropDuplicates()
        # Drop duplicates that are exact duplicates apart from ID
        df = df.dropDuplicates(subset=[c for c in df.columns if c != self._id_col])
        if self._id_col == '' or self._id_col is None:
            return df
        # Provide monotonically increasing, unique ID (in case of duplicate ID)
        df = df.withColumn(self._id_col, fn.monotonically_increasing_id())
        return df