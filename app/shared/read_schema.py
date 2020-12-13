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