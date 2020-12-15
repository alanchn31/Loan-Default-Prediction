from enum import Enum
from typing import Callable
from pyspark.sql import DataFrame
from pyspark.ml import Transformer, PipelineModel
from typing import List
from pipe.pipe import Pipe


class IF(Transformer):
    """
    Conditional pipeline which runs one or another list of transformers based on condition
    
    original author: https://github.com/bbiletskyy/pipeline-oriented-analytics
    """

    class Predicate:
        @classmethod
        def const(cls, value: bool) -> Callable[[DataFrame], bool]:
            def predicate(df: DataFrame) -> bool:
                return value
            return predicate

        @classmethod
        def has_column(cls, name: str) -> Callable[[DataFrame], bool]:
            def predicate(df: DataFrame) -> bool:

                return name in set(df.columns)

            return predicate

    @classmethod
    def condition(cls, condition: bool) -> Callable[[DataFrame], bool]:
        def constant_predicate(df: DataFrame) -> bool:
            return condition
        return constant_predicate

    @classmethod
    def has_columns(cls, columns: List[str]) -> Callable[[DataFrame], bool]:
        def predicate(df: DataFrame) -> bool:
            return set(columns).issubset(df.columns)

        return predicate()

    def __init__(self, condition: Callable[[DataFrame], bool],
                 then: List[Transformer], otherwise: List[Transformer] = None):
        super(IF, self).__init__()
        self._condition = condition
        self._then = Pipe(then)
        self._otherwise = Pipe([])
        if otherwise is not None:
            self._otherwise = Pipe(otherwise)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self._condition(dataset) is True:
            return self._then.transform(dataset)
        else:
            return self._otherwise.transform(dataset)