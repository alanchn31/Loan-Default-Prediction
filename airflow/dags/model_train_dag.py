from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator

try:
    from airflow.operators import LivyBatchOperator
except ImportError:
    from airflow_livy.batch import LivyBatchOperator

# Define configured variables to connect to AWS S3 and Redshift
aws_config = Variable.get("aws_config", deserialize_json=True)
main_file_path = Variable.get("main_file_path")
pyfiles_path = Variable.get("pyfiles_path")
config_file_path = Variable.get("config_file_path")

# Default settings for DAG
default_args = {
    'owner': 'Alan',
    'depends_on_past': False,
    'start_date': datetime.today(),
    'retries': 5,
    'retry_delay': timedelta(minutes=1),
}

with DAG(dag_id='loan_prediction_train_model', default_args=default_args,
         description='"Run Spark job via Livy Batches \
                      to preprocess data and train model',
         schedule_interval='@once') as dag:

    preprocess_data_step = LivyBatchOperator(
        name="preprocess_data_{{ run_id }}",
        file=main_file_path,
        py_files=[pyfiles_path],
        files=[config_file_path],
        arguments=[
            "--job", "preprocess_data",
            "--phase", "train",
            "--mode", "aws",
            "--awsKey", aws_config['awsKey'],
            "--awsSecretKey", aws_config['awsSecretKey'],
            "--s3Bucket", aws_config['s3Bucket']
        ],
        task_id="preprocess_data"
    )

    train_model_step = LivyBatchOperator(
        name="train_model_{{ run_id }}",
        file=main_file_path,
        py_files=[pyfiles_path],
        arguments=[
            "--job", "preprocess_data",
            "--phase", "train",
            "--mode", "aws",
            "--awsKey", aws_config['awsKey'],
            "--awsSecretKey", aws_config['awsSecretKey'],
            "--s3Bucket", aws_config['s3Bucket']
        ],
        task_id="train_model"
    )

    preprocess_data_step >> train_model_step