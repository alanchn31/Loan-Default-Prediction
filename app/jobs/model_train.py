import json
import pyspark.sql.functions as fn
import pyspark.ml.feature as sf
from pyspark.sql import SparkSession
from app.shared.read_schema import read_schema


def _extract_data(spark, config):
    schema_lst = config.get("schema")
    schema = read_schema(schema_lst)
    file_path = config.get('source_data_path')
    return spark.read.csv(file_path,
                          header=True,
                          schema=schema)


def _remove_duplicates(df):
    # Drop exact duplicates
    df = df.dropDuplicates()
    # Drop duplicates that are exact duplicates apart from ID
    df = df.dropDuplicates(subset=[c for c in df.columns if c != 'ID'])
    # Provide monotonically increasing, unique ID (in case of duplicate ID)
    df = df.withColumn('ID', fn.monotonically_increasing_id())
    return df


def _impute_missing_values(df, config):
    impute_dict = config.get("impute_cols")
    for col, val in impute_dict.items():
        # Specify mean/median for numerical cols
        if val == "mean" or val == "median":
            imputer = sf.Imputer(
                inputCols=[col], 
                outputCols=[col],
                strategy=val
            )
            df = imputer.fit(df).transform(df)
        # Specify custom value for categorical cols
        else:
            df = df.fillna({col: val})
    return df


def run_job(spark, config):
    """ Runs model training job"""
    raw_df = _extract_data(spark, config)
    df = _remove_duplicates(raw_df)
    df = _impute_missing_values(df)
    # Print output as a test, will remove later
    print(raw_df.take(5))