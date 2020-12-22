from airflow.plugins_manager import AirflowPlugin

from .batch import LivyBatchOperator, LivyBatchSensor
from .session import (
    LivySessionCreationSensor,
    LivySessionOperator,
    LivyStatementSensor,
)


class LivyBatchPlugin(AirflowPlugin):
    name = "livy_batch_plugin"
    sensors = [LivyBatchSensor]
    operators = [LivyBatchOperator]


class LivySessionPlugin(AirflowPlugin):
    name = "livy_session_plugin"
    sensors = [LivySessionCreationSensor, LivyStatementSensor]
    operators = [LivySessionOperator]