from pyspark.sql import DataFrame
from pyspark.ml import Transformer, PipelineModel
from typing import List


class Pipe(Transformer):
    """
    Conditional pipeline which runs one or another list of transformers based on condition

    original author: https://github.com/bbiletskyy/pipeline-oriented-analytics
    """

    def __init__(self, stages: List[Transformer]):
        super(Pipe, self).__init__()
        self._pipeline = PipelineModel(stages)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return self._pipeline.transform(dataset)