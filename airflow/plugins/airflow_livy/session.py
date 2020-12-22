"""
From: https://github.com/panovvv/airflow-livy-operators/blob/master/airflow_home/plugins/airflow_livy/session.py
"""

import json
import logging
from json import JSONDecodeError
from numbers import Number
from typing import List

from airflow.exceptions import AirflowBadRequest, AirflowException
from airflow.hooks.http_hook import HttpHook
from airflow.models import BaseOperator
from airflow.sensors.base_sensor_operator import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

ENDPOINT = "sessions"
ALLOWED_LANGUAGES = ["spark", "pyspark", "sparkr", "sql"]
LOG_PAGE_LINES = 100


def log_response_error(lookup_path, response, session_id=None, statement_id=None):
    msg = "Can not parse JSON response."
    if session_id is not None:
        msg += f" Session id={session_id}."
    if statement_id is not None:
        msg += f" Statement id={statement_id}."
    try:
        pp_response = (
            json.dumps(json.loads(response.content), indent=2)
            if "application/json" in response.headers.get("Content-Type", "")
            else response.content
        )
    except AttributeError:
        pp_response = json.dumps(response, indent=2)
    msg += f"\nTried to find JSON path: {lookup_path}, but response was:\n{pp_response}"
    logging.error(msg)


def validate_timings(poke_interval, timeout):
    if poke_interval < 1:
        raise AirflowException(
            f"Poke interval {poke_interval} sec. is too small, "
            f"this will result in too frequent API calls"
        )
    if poke_interval > timeout:
        raise AirflowException(
            f"Poke interval {poke_interval} sec. is greater "
            f"than the timeout value {timeout} sec. Timeout won't work."
        )


class LivySessionCreationSensor(BaseSensorOperator):
    def __init__(
        self,
        session_id,
        poke_interval,
        timeout,
        task_id,
        http_conn_id="livy",
        soft_fail=False,
        mode="poke",
    ):
        validate_timings(poke_interval, timeout)
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            soft_fail=soft_fail,
            mode=mode,
            task_id=task_id,
        )
        self.session_id = session_id
        self.http_conn_id = http_conn_id

    def poke(self, context):
        logging.info(f"Getting session {self.session_id} status...")
        endpoint = f"{ENDPOINT}/{self.session_id}/state"
        response = HttpHook(method="GET", http_conn_id=self.http_conn_id).run(endpoint)
        try:
            state = json.loads(response.content)["state"]
        except (JSONDecodeError, LookupError) as ex:
            log_response_error("$.state", response, self.session_id)
            raise AirflowBadRequest(ex)
        if state == "starting":
            logging.info(f"Session {self.session_id} is starting...")
            return False
        if state == "idle":
            logging.info(f"Session {self.session_id} is ready to receive statements.")
            return True
        raise AirflowException(
            f"Session {self.session_id} failed to start. "
            f"State='{state}'. Expected states: 'starting' or 'idle' (ready)."
        )


class LivyStatementSensor(BaseSensorOperator):
    def __init__(
        self,
        session_id,
        statement_id,
        poke_interval,
        timeout,
        task_id,
        http_conn_id="livy",
        soft_fail=False,
        mode="poke",
    ):
        validate_timings(poke_interval, timeout)
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            soft_fail=soft_fail,
            mode=mode,
            task_id=task_id,
        )
        self.session_id = session_id
        self.statement_id = statement_id
        self.http_conn_id = http_conn_id

    def poke(self, context):
        logging.info(
            f"Getting status for statement {self.statement_id} "
            f"in session {self.session_id}"
        )
        endpoint = f"{ENDPOINT}/{self.session_id}/statements/{self.statement_id}"
        response = HttpHook(method="GET", http_conn_id=self.http_conn_id).run(endpoint)
        try:
            statement = json.loads(response.content)
            state = statement["state"]
        except (JSONDecodeError, LookupError) as ex:
            log_response_error("$.state", response, self.session_id, self.statement_id)
            raise AirflowBadRequest(ex)
        if state in ["waiting", "running"]:
            logging.info(
                f"Statement {self.statement_id} in session {self.session_id} "
                f"has not finished yet (state is '{state}')"
            )
            return False
        if state == "available":
            self.__check_status(statement, response)
            return True
        raise AirflowBadRequest(
            f"Statement {self.statement_id} in session {self.session_id} failed due to "
            f"an unknown state: '{state}'.\nKnown states: 'waiting', 'running', "
            "'available'"
        )

    def __check_status(self, statement, response):
        try:
            output = statement["output"]
            status = output["status"]
        except LookupError as ex:
            log_response_error(
                "$.output.status", response, self.session_id, self.statement_id
            )
            raise AirflowBadRequest(ex)
        pp_output = "\n".join(json.dumps(output, indent=2).split("\\n"))
        logging.info(
            f"Statement {self.statement_id} in session {self.session_id} "
            f"finished. Output:\n{pp_output}"
        )
        if status != "ok":
            raise AirflowBadRequest(
                f"Statement {self.statement_id} in session {self.session_id} "
                f"failed with status '{status}'. Expected status is 'ok'"
            )


class LivySessionOperator(BaseOperator):
    class Statement:
        template_fields = ["code"]
        code: str
        kind: str

        def __init__(self, code, kind=None):
            if kind in ALLOWED_LANGUAGES or kind is None:
                self.kind = kind
            else:
                raise AirflowException(
                    f"Can not create statement with kind '{kind}'!\n"
                    f"Allowed session kinds: {ALLOWED_LANGUAGES}"
                )
            self.code = code

        def __str__(self) -> str:
            dashes = 80
            return (
                f"\n{{\n  Statement, kind: {self.kind}"
                f"\n  code:\n{'-'*dashes}\n{self.code}\n{'-'*dashes}\n}}"
            )

        __repr__ = __str__

    template_fields = [
        "name",
        "statements",
    ]

    @apply_defaults
    def __init__(
        self,
        statements: List[Statement],
        kind: str = None,
        proxy_user=None,
        jars=None,
        py_files=None,
        files=None,
        driver_memory=None,
        driver_cores=None,
        executor_memory=None,
        executor_cores=None,
        num_executors=None,
        archives=None,
        queue=None,
        name=None,
        conf=None,
        heartbeat_timeout=None,
        session_start_timeout_sec=120,
        session_start_poll_period_sec=10,
        statemt_timeout_minutes=10,
        statemt_poll_period_sec=20,
        http_conn_id="livy",
        spill_logs=False,
        *args,
        **kwargs,
    ):
        super(LivySessionOperator, self).__init__(*args, **kwargs)
        if kind in ALLOWED_LANGUAGES or kind is None:
            self.kind = kind
        else:
            raise AirflowException(
                f"Can not create session with kind '{kind}'!\n"
                f"Allowed session kinds: {ALLOWED_LANGUAGES}"
            )
        self.statements = statements
        self.proxy_user = proxy_user
        self.jars = jars
        self.py_files = py_files
        self.files = files
        self.driver_memory = driver_memory
        self.driver_cores = driver_cores
        self.executor_memory = executor_memory
        self.executor_cores = executor_cores
        self.num_executors = num_executors
        self.archives = archives
        self.queue = queue
        self.name = name
        self.conf = conf
        self.heartbeat_timeout = heartbeat_timeout
        self.session_start_timeout_sec = session_start_timeout_sec
        self.session_start_poll_period_sec = session_start_poll_period_sec
        self.statemt_timeout_minutes = statemt_timeout_minutes
        self.statemt_poll_period_sec = statemt_poll_period_sec
        self.http_conn_id = http_conn_id
        self.spill_logs = spill_logs
        self.session_id = None

    def execute(self, context):
        try:
            self.create_session()
            logging.info(f"Session has been created with id = {self.session_id}.")
            LivySessionCreationSensor(
                self.session_id,
                task_id=self.task_id,
                http_conn_id=self.http_conn_id,
                poke_interval=self.session_start_poll_period_sec,
                timeout=self.session_start_timeout_sec,
            ).execute(context)
            logging.info(f"Session {self.session_id} is ready to accept statements.")
            for i, statement in enumerate(self.statements):
                logging.info(
                    f"Submitting statement {i+1}/{len(self.statements)} "
                    f"in session {self.session_id}..."
                )
                statement_id = self.submit_statement(statement)
                logging.info(
                    f"Statement {i+1}/{len(self.statements)} "
                    f"(session {self.session_id}) "
                    f"has been submitted with id {statement_id}"
                )
                LivyStatementSensor(
                    self.session_id,
                    statement_id,
                    task_id=self.task_id,
                    http_conn_id=self.http_conn_id,
                    poke_interval=self.statemt_poll_period_sec,
                    timeout=self.statemt_timeout_minutes * 60,
                ).execute(context)
            logging.info(
                f"All {len(self.statements)} statements in session {self.session_id} "
                f"completed successfully!"
            )
        except Exception:
            if self.session_id is not None:
                self.spill_session_logs()
                self.spill_logs = False
            raise
        finally:
            if self.session_id is not None:
                if self.spill_logs:
                    self.spill_session_logs()
                self.close_session()

    def create_session(self):
        headers = {"X-Requested-By": "airflow", "Content-Type": "application/json"}
        unfiltered_payload = {
            "kind": self.kind,
            "proxyUser": self.proxy_user,
            "jars": self.jars,
            "pyFiles": self.py_files,
            "files": self.files,
            "driverMemory": self.driver_memory,
            "driverCores": self.driver_cores,
            "executorMemory": self.executor_memory,
            "executorCores": self.executor_cores,
            "numExecutors": self.num_executors,
            "archives": self.archives,
            "queue": self.queue,
            "name": self.name,
            "conf": self.conf,
            "heartbeatTimeoutInSecond": self.heartbeat_timeout,
        }
        payload = {k: v for k, v in unfiltered_payload.items() if v}
        logging.info(
            f"Creating a session in Livy... "
            f"Payload:\n{json.dumps(payload, indent=2)}"
        )
        response = HttpHook(http_conn_id=self.http_conn_id).run(
            ENDPOINT, json.dumps(payload), headers,
        )
        try:
            session_id = json.loads(response.content)["id"]
        except (JSONDecodeError, LookupError) as ex:
            log_response_error("$.id", response)
            raise AirflowBadRequest(ex)

        if not isinstance(session_id, Number):
            raise AirflowException(
                f"ID of the created session is not a number ({session_id}). "
                "Are you sure we're calling Livy API?"
            )
        self.session_id = session_id

    def submit_statement(self, statement: Statement):
        headers = {"X-Requested-By": "airflow", "Content-Type": "application/json"}
        payload = {"code": statement.code}
        if statement.kind:
            payload["kind"] = statement.kind
        endpoint = f"{ENDPOINT}/{self.session_id}/statements"
        response = HttpHook(http_conn_id=self.http_conn_id).run(
            endpoint, json.dumps(payload), headers
        )
        try:
            return json.loads(response.content)["id"]
        except (JSONDecodeError, LookupError) as ex:
            log_response_error("$.id", response, self.session_id)
            raise AirflowBadRequest(ex)

    def spill_session_logs(self):
        dashes = 50
        logging.info(f"{'-'*dashes}Full log for session {self.session_id}{'-'*dashes}")
        endpoint = f"{ENDPOINT}/{self.session_id}/log"
        hook = HttpHook(method="GET", http_conn_id=self.http_conn_id)
        line_from = 0
        line_to = LOG_PAGE_LINES
        while True:
            log_page = self.fetch_log_page(hook, endpoint, line_from, line_to)
            try:
                logs = log_page["log"]
                for log in logs:
                    logging.info(log.replace("\\n", "\n"))
                actual_line_from = log_page["from"]
                total_lines = log_page["total"]
            except LookupError as ex:
                log_response_error("$.log, $.from, $.total", log_page, self.session_id)
                raise AirflowBadRequest(ex)
            actual_lines = len(logs)
            if actual_line_from + actual_lines >= total_lines:
                logging.info(
                    f"{'-' * dashes}End of full log for session {self.session_id}"
                    f"{'-' * dashes}"
                )
                break
            line_from = actual_line_from + actual_lines

    @staticmethod
    def fetch_log_page(hook: HttpHook, endpoint, line_from, line_to):
        prepd_endpoint = endpoint + f"?from={line_from}&size={line_to}"
        response = hook.run(prepd_endpoint)
        try:
            return json.loads(response.content)
        except JSONDecodeError as ex:
            log_response_error("$", response)
            raise AirflowBadRequest(ex)

    def close_session(self):
        logging.info(f"Closing session with id = {self.session_id}")
        session_endpoint = f"{ENDPOINT}/{self.session_id}"
        HttpHook(method="DELETE", http_conn_id=self.http_conn_id).run(session_endpoint)
        logging.info(f"Session {self.session_id} has been closed")