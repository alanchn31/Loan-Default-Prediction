"""
From: https://github.com/panovvv/airflow-livy-operators/blob/master/airflow_home/plugins/airflow_livy/batch.py
"""

import json
import logging
from json import JSONDecodeError
from numbers import Number

from airflow.exceptions import AirflowBadRequest, AirflowException
from airflow.hooks.http_hook import HttpHook
from airflow.models import BaseOperator
from airflow.sensors.base_sensor_operator import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

LIVY_ENDPOINT = "batches"
SPARK_ENDPOINT = "api/v1/applications"
YARN_ENDPOINT = "ws/v1/cluster/apps"
VERIFICATION_METHODS = ["spark", "yarn"]
VALID_BATCH_STATES = [
    "not_started",
    "starting",
    "recovering",
    "idle",
    "running",
    "busy",
    "shutting_down",
]
LOG_PAGE_LINES = 100


def log_response_error(lookup_path, response, batch_id=None):
    msg = "Can not parse JSON response."
    if batch_id is not None:
        msg += f" Batch id={batch_id}."
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


class LivyBatchSensor(BaseSensorOperator):
    def __init__(
        self,
        batch_id,
        task_id,
        poke_interval=20,
        timeout=10 * 60,
        http_conn_id="livy",
        soft_fail=False,
        mode="poke",
    ):
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
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            soft_fail=soft_fail,
            mode=mode,
            task_id=task_id,
        )
        self.batch_id = batch_id
        self.http_conn_id = http_conn_id

    def poke(self, context):
        logging.info(f"Getting batch {self.batch_id} status...")
        endpoint = f"{LIVY_ENDPOINT}/{self.batch_id}"
        response = HttpHook(method="GET", http_conn_id=self.http_conn_id).run(endpoint)
        try:
            state = json.loads(response.content)["state"]
        except (JSONDecodeError, LookupError) as ex:
            log_response_error("$.state", response, self.batch_id)
            raise AirflowBadRequest(ex)
        if state in VALID_BATCH_STATES:
            logging.info(
                f"Batch {self.batch_id} has not finished yet (state is '{state}')"
            )
            return False
        if state == "success":
            logging.info(f"Batch {self.batch_id} has finished successfully!")
            return True
        raise AirflowException(f"Batch {self.batch_id} failed with state '{state}'")


class LivyBatchOperator(BaseOperator):
    template_fields = ["name", "arguments"]

    @apply_defaults
    def __init__(
        self,
        file=None,
        proxy_user=None,
        class_name=None,
        arguments=None,
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
        timeout_minutes=10,
        poll_period_sec=20,
        verify_in=None,
        http_conn_id_livy="livy",
        http_conn_id_spark="spark",
        http_conn_id_yarn="yarn",
        spill_logs=True,
        *args,
        **kwargs,
    ):
        super(LivyBatchOperator, self).__init__(*args, **kwargs)
        self.file = file
        self.proxy_user = proxy_user
        self.class_name = class_name
        self.arguments = arguments
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
        self.timeout_minutes = timeout_minutes
        self.poll_period_sec = poll_period_sec
        if verify_in in VERIFICATION_METHODS or verify_in is None:
            self.verify_in = verify_in
        else:
            raise AirflowException(
                f"Can not create batch operator with verification method "
                f"'{verify_in}'!\nAllowed methods: {VERIFICATION_METHODS}"
            )
        self.http_conn_id_livy = http_conn_id_livy
        self.http_conn_id_spark = http_conn_id_spark
        self.http_conn_id_yarn = http_conn_id_yarn
        self.spill_logs = spill_logs
        self.batch_id = None

    def execute(self, context):
        try:
            self.submit_batch()
            logging.info(f"Batch successfully submitted with id = {self.batch_id}.")
            LivyBatchSensor(
                self.batch_id,
                task_id=self.task_id,
                http_conn_id=self.http_conn_id_livy,
                poke_interval=self.poll_period_sec,
                timeout=self.timeout_minutes * 60,
            ).execute(context)
            if self.verify_in in VERIFICATION_METHODS:
                logging.info(
                    f"Additionally verifying status for batch id {self.batch_id} "
                    f"via {self.verify_in}..."
                )
                self.verify()
        except Exception:
            if self.batch_id is not None:
                self.spill_batch_logs()
                self.close_batch()
                self.batch_id = None
            raise
        finally:
            if self.batch_id is not None:
                if self.spill_logs:
                    self.spill_batch_logs()
                self.close_batch()

    def submit_batch(self):
        headers = {"X-Requested-By": "airflow", "Content-Type": "application/json"}
        unfiltered_payload = {
            "file": self.file,
            "proxyUser": self.proxy_user,
            "className": self.class_name,
            "args": self.arguments,
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
        }
        payload = {k: v for k, v in unfiltered_payload.items() if v}
        logging.info(
            f"Submitting the batch to Livy... "
            f"Payload:\n{json.dumps(payload, indent=2)}"
        )
        response = HttpHook(http_conn_id=self.http_conn_id_livy).run(
            LIVY_ENDPOINT, json.dumps(payload), headers
        )
        try:
            batch_id = json.loads(response.content)["id"]
        except (JSONDecodeError, LookupError) as ex:
            log_response_error("$.id", response)
            raise AirflowBadRequest(ex)
        if not isinstance(batch_id, Number):
            raise AirflowException(
                f"ID of the created batch is not a number ({batch_id}). "
                "Are you sure we're calling Livy API?"
            )
        self.batch_id = batch_id

    def verify(self):
        app_id = self.get_spark_app_id(self.batch_id)
        if app_id is None:
            raise AirflowException(f"Spark appId was null for batch {self.batch_id}")
        logging.info(f"Found app id '{app_id}' for batch id {self.batch_id}.")
        if self.verify_in == "spark":
            self.check_spark_app_status(app_id)
        else:
            self.check_yarn_app_status(app_id)
        logging.info(f"App '{app_id}' associated with batch {self.batch_id} completed!")

    def get_spark_app_id(self, batch_id):
        logging.info(f"Getting Spark app id from Livy API for batch {batch_id}...")
        endpoint = f"{LIVY_ENDPOINT}/{batch_id}"
        response = HttpHook(method="GET", http_conn_id=self.http_conn_id_livy).run(
            endpoint
        )
        try:
            return json.loads(response.content)["appId"]
        except (JSONDecodeError, LookupError, AirflowException) as ex:
            log_response_error("$.appId", response, batch_id)
            raise AirflowBadRequest(ex)

    def check_spark_app_status(self, app_id):
        logging.info(f"Getting app status (id={app_id}) from Spark REST API...")
        endpoint = f"{SPARK_ENDPOINT}/{app_id}/jobs"
        response = HttpHook(method="GET", http_conn_id=self.http_conn_id_spark).run(
            endpoint
        )
        try:
            jobs = json.loads(response.content)
            expected_status = "SUCCEEDED"
            for job in jobs:
                job_id = job["jobId"]
                job_status = job["status"]
                logging.info(
                    f"Job id {job_id} associated with application '{app_id}' "
                    f"is '{job_status}'"
                )
                if job_status != expected_status:
                    raise AirflowException(
                        f"Job id '{job_id}' associated with application '{app_id}' "
                        f"is '{job_status}', expected status is '{expected_status}'"
                    )
        except (JSONDecodeError, LookupError, TypeError) as ex:
            log_response_error("$.jobId, $.status", response)
            raise AirflowBadRequest(ex)

    def check_yarn_app_status(self, app_id):
        logging.info(f"Getting app status (id={app_id}) from YARN RM REST API...")
        endpoint = f"{YARN_ENDPOINT}/{app_id}"
        response = HttpHook(method="GET", http_conn_id=self.http_conn_id_yarn).run(
            endpoint
        )
        try:
            status = json.loads(response.content)["app"]["finalStatus"]
        except (JSONDecodeError, LookupError, TypeError) as ex:
            log_response_error("$.app.finalStatus", response)
            raise AirflowBadRequest(ex)
        expected_status = "SUCCEEDED"
        if status != expected_status:
            raise AirflowException(
                f"YARN app {app_id} is '{status}', expected status: '{expected_status}'"
            )

    def spill_batch_logs(self):
        dashes = 50
        logging.info(f"{'-'*dashes}Full log for batch {self.batch_id}{'-'*dashes}")
        endpoint = f"{LIVY_ENDPOINT}/{self.batch_id}/log"
        hook = HttpHook(method="GET", http_conn_id=self.http_conn_id_livy)
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
                log_response_error("$.log, $.from, $.total", log_page)
                raise AirflowBadRequest(ex)
            actual_lines = len(logs)
            if actual_line_from + actual_lines >= total_lines:
                logging.info(
                    f"{'-' * dashes}End of full log for batch {self.batch_id}"
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

    def close_batch(self):
        logging.info(f"Closing batch with id = {self.batch_id}")
        batch_endpoint = f"{LIVY_ENDPOINT}/{self.batch_id}"
        HttpHook(method="DELETE", http_conn_id=self.http_conn_id_livy).run(
            batch_endpoint
        )
        logging.info(f"Batch {self.batch_id} has been closed")