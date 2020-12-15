import json
import pyspark.sql.functions as fn
import pyspark.sql.types as typ
import pyspark.ml.feature as sf
from pyspark.sql import SparkSession
from shared.utils import read_schema


def _read_data(spark, config):
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


def _check_numerical_dtype(df, column, cast_float=False):
    df_data_types = dict(df.dtypes)
    accepted_dtypes = ['int', 'bigint', 'decimal', 'double', 'float', 
                       'long', 'bigdecimal', 'byte', 'short']
    if df_data_types[column] not in accepted_dtypes:
        raise Exception("Non-numeric value found in column")
    if cast_float:
        if df_data_types[column] not in ['double', 'float']:
            df = df.withColumn(column, df[column].cast(typ.DoubleType()))
    return df


def _impute_missing_values(df, config):
    impute_dict = config.get("impute_cols", None)
    if impute_dict == {} or impute_dict is None:
        return df
    for column, val in impute_dict.items():
        # Specify mean/median for numerical cols
        if val == "mean" or val == "median":
            df = _check_numerical_dtype(df, column, cast_float=True)
            imputer = sf.Imputer(
                inputCols=[column], 
                outputCols=[column],
                strategy=val
            )
            df = imputer.fit(df).transform(df)
        # Specify custom value for categorical cols
        else:
            df = df.fillna({column: val})
    return df


def _impute_outliers(df, config):
    """
    Impute outliers by winsorizing selected numerical columns 
    (clips to (1st/3rd quantile) -+ 1.5*IQR)
    """
    winsorize_cols = config.get("winsorize_cols", None)
    if winsorize_cols == [] or winsorize_cols is None:
        return df
    bounds = {}
    for wc in winsorize_cols:
        df = _check_numerical_dtype(df, wc)
        quantiles = df.approxQuantile(
                        wc, [0.25, 0.75], 0.05
                    )
        iqr = quantiles[1] - quantiles[0]
        bounds[wc] = [quantiles[0] - 1.5 * iqr,
                       quantiles[1] + 1.5 * iqr]
        df = df.withColumn(wc, fn.when(fn.col(wc) < bounds[wc][0], bounds[wc][0])
                                 .when(fn.col(wc) > bounds[wc][1], bounds[wc][1])
                                 .otherwise(fn.col(wc))
        )
    return df


def _convert_str_to_date(df, config):
    """
    Converts date string column to date column
    """
    date_str_cols = config.get("date_str_cols", None)
    if date_str_cols == {} or date_str_cols is None:
        return df
    
    # Converts "1/1/99" --> "01/01/1999", then convert to date object
    for column, year_prefix in date_str_cols.items():
        df = df.withColumn(column, fn.to_date(column, "d/M/yy"))
        df = df.withColumn(column, fn.when(fn.year(column) > (int(year_prefix) + 1)*100, 
                                           fn.add_months(column, -12*100)) \
                                     .when(fn.year(column) == 1900, 
                                           fn.add_months(column, 12*100)) \
                                     .otherwise(fn.col(column)))
            
    return df


def _get_age(df, config):
    start_col = config["age_cols"]["start"]
    end_col = config["age_cols"]["end"]
    df = df.withColumn(config["age_cols"]["output_col"], 
                       fn.floor(fn.datediff(end_col, start_col)/365.25))
    return df


def _extract_time_period_mths(df, config):
    """
    Finds the number of months from a time period string
    for eg: 2yrs 2mon --> 26
    """
    tenure_cols = config.get("tenure_cols", None)
    if tenure_cols == [] or tenure_cols is None:
        return df
    for column in tenure_cols:
        df = df.withColumn(f"{column}_YRS", fn.regexp_extract(column, "(\d)(yrs)", 1))
        df = df.withColumn(f"{column}_MTHS", fn.regexp_extract(column, "(\d)(mon)", 1))
        df = df.withColumn(column, (fn.col(f"{column}_YRS")*12 + fn.col(f"{column}_MTHS")))
        df = df.drop(*[f"{column}_YRS", f"{column}_MTHS"]) 
    return df


def _replace_str_regex(df, config):
    """
    Replace string in a String column using regex
    """
    str_replace_cols = config.get("str_replace_cols", None)
    if str_replace_cols == [] or str_replace_cols is None:
        return df
    for column, val in str_replace_cols.items():
        pattern = val["pattern"]
        replacement = val["replacement"]
        df = df.withColumn(column, fn.regexp_replace(column, pattern, replacement))
    return df


def _drop_cols(df, config):
    """
    drop columns that are not needed
    """
    drop_cols = config.get("drop_cols", None)
    if drop_cols == [] or drop_cols is None:
        return df
    df = df.drop(*drop_cols)
    return df


def run_job(spark, config):
    """ Runs model training job"""
    raw_df = _read_data(spark, config)
    df = _remove_duplicates(raw_df)
    df = _impute_missing_values(df, config)
    df = _impute_outliers(df, config)
    df = _convert_str_to_date(df, config)
    df = _get_age(df, config)
    df = _extract_time_period_mths(df, config)
    df = _replace_str_regex(df, config)
    df = _drop_cols(df, config)
    return df