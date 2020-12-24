import json
import pyspark.sql.functions as fn
import pyspark.sql.types as typ
import pyspark.ml.feature as sf
from pipe.pipe import Pipe
from pipe.IF import IF
from transformers.remove_duplicates import RemoveDuplicates
from transformers.convert_str_to_date import ConvertStrToDate
from transformers.get_age import GetAge
from transformers.extract_time_period_mths import ExtractTimePeriodMths
from transformers.replace_str_regex import ReplaceStrRegex
from transformers.drop_columns import DropColumns
from transformers.impute_cat_missing_vals import ImputeCategoricalMissingVals
from pyspark.sql import SparkSession
from shared.utils import read_schema


def _read_data(spark, config, mode, phase, s3_bucket):
    schema_lst = config.get("schema")
    schema = read_schema(schema_lst)
    if mode == "local":
        file_path = config.get('source_data_path')
    else:
        file_path = config.get('s3_source_data_path').format(s3_bucket)
    if phase == "train":
        file_path += config.get('train_data_file')
    else:
        file_path += config.get('pred_data_file')
    return spark.read.csv(file_path,
                          header=True,
                          schema=schema)


def run_job(spark, config, mode, s3_bucket=None, phase=None):
    """ Runs Data Preparation job"""
    raw_df = _read_data(spark, config, mode, phase, s3_bucket)
    phase = 'train' if "DEFAULT" in raw_df.columns else 'test'
    df = Pipe([
        IF(IF.Predicate.has_column('DEFAULT'), then=[
            RemoveDuplicates(config['id_col'])
        ]),
        ConvertStrToDate(config['date_str_cols']),
        GetAge(config['age_cols']),
        ExtractTimePeriodMths(config['tenure_cols']),
        ReplaceStrRegex(config['str_replace_cols']),
        ImputeCategoricalMissingVals(config['impute_cat_cols']),
        DropColumns(config['drop_cols']),
    ]).transform(raw_df)
    if mode == "local":
        df.write.parquet(config['processed_data_dir'] + f"{phase}.parquet", mode='overwrite')
    else:
        df.write.parquet(config['s3_processed_data_dir'].format(s3_bucket) + f"{phase}.parquet", 
                        mode='overwrite')