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


def create_spark_session(config, mode, aws_key, aws_secret_key):
    """
    Description: Creates spark session.
    Returns:
        spark session object
    """
    if mode == "local":
         spark = SparkSession.builder \
                        .appName(config.get("app_name")) \
                        .getOrCreate()
    else:
        spark = SparkSession.builder \
                            .config("spark.executor.heartbeatInterval", "40s") \
                            .appName(config.get("app_name")) \
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

    spark_files_dir = SparkFiles.getRootDirectory()
    config_files = [filename
                    for filename in listdir(spark_files_dir)
                    if filename.endswith('config.json')]

    if config_files:
        path_to_config_file = path.join(spark_files_dir, config_files[0])
        with open(path_to_config_file, 'r') as config_file:
            config = json.load(config_file)
        spark_logger.warn('loaded config from ' + config_files[0])
    else:
        spark_logger.warn('no config file found')
        config = None

    spark = create_spark_session(config, args.mode, args.awsKey, args.awsSecretKey)

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