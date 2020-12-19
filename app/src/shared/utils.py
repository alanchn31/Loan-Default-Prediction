import pyspark.sql.types as typ


def read_schema(schema_arg):
    d_types = {
        "StringType()": typ.StringType(),
        "IntegerType()": typ.IntegerType(),
        "TimestampType()": typ.TimestampType(),
        "DoubleType()": typ.DoubleType()
    }

    schema = typ.StructType()
    for s in schema_arg:
        x = s.split(" ")
        schema.add(x[0], d_types[x[1]], True)
    return schema


def check_numerical_dtype(df, column, cast_float=False):
    df_data_types = dict(df.dtypes)
    accepted_dtypes = ['int', 'bigint', 'decimal', 'double', 'float', 
                       'long', 'bigdecimal', 'byte', 'short']
    if df_data_types[column] not in accepted_dtypes:
        raise Exception("Non-numeric value found in column")