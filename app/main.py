import os
import sys
import json
import importlib
import argparse
from pyspark import SparkFiles
from pyspark.sql import SparkSession


def _parse_arguments():
    """
    Parse arguments provided by spark-submit command
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase", required=False)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--awsKey", required=False)
    parser.add_argument("--awsSecretKey", required=False)
    parser.add_argument("--s3Bucket", required=False)
    return parser.parse_args()


def create_spark_session(mode, aws_key, aws_secret_key):
    """
    Description: Creates spark session.
    Returns:
        spark session object
    """
    if mode == "local":
         spark = SparkSession.builder \
                        .appName("LoanDefaultPrediction") \
                        .getOrCreate()
    else:
        spark = SparkSession.builder \
                            .config("spark.executor.heartbeatInterval", "40s") \
                            .appName("LoanDefaultPrediction") \
                            .getOrCreate()
        
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl",
                                                        "org.apache.hadoop.fs.s3a.S3AFileSystem")
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", aws_key)
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", aws_secret_key)
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.connection.timeout", "1000")
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.connection.maximum", "5000")
    return spark


def main():
    """
    Main function executed by spark-submit command
    """
    args = _parse_arguments()

    spark = create_spark_session(args.mode, args.awsKey, args.awsSecretKey)
    log4j = spark._jvm.org.apache.log4j
    conf = spark.sparkContext.getConf()
    app_id = conf.get('spark.app.id')
    app_name = conf.get('spark.app.name')
    message_prefix = '<' + app_name + ' ' + app_id + '>'
    spark_logger = log4j.LogManager.getLogger(message_prefix)

    spark_files_dir = SparkFiles.getRootDirectory()
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    config_files = [filename
                    for filename in os.listdir(spark_files_dir)
                    if filename.endswith('config.json')]

    if config_files:
        path_to_config_file = os.path.join(spark_files_dir, config_files[0])
        with open(path_to_config_file, 'r') as config_file:
            config = json.load(config_file)
        spark_logger.warn('loaded config from ' + config_files[0])
    else:
        spark_logger.warn('no config file found')
        config = None

    job_module = importlib.import_module(f"jobs.{args.job}")
    if args.phase:
        job_module.run_job(spark, config, args.mode, args.s3Bucket, args.phase)
    else:
        job_module.run_job(spark, config, args.mode)


if os.path.exists('src.zip'):
    sys.path.insert(0, 'src.zip')
else:
    sys.path.insert(0, './src')


if __name__ == "__main__":
    main()