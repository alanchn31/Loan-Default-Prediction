from pyspark.ml import Transformer
from pyspark.sql import DataFrame


class SaveToParquet(Transformer):
    """
    Saves dataframe to parquet files
    """

    def __init__(self, path: str):
        super(SaveToParquet, self).__init__()
        self._path = path

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset.write.parquet(self._path, mode='overwrite')
        return dataset