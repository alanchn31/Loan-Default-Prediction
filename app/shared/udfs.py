import pyspark.sql.types as typ
from pyspark.sql.functions import (col, to_date, unix_timestamp, 
                                  regexp_replace, udf)


def get_formatted_date(date_str, year_prefix):
    date_components = date_str.split("/")
    day = date_components[0].zfill(2)
    month = date_components[1].zfill(2)
    if date_components[2] == "00":
        year = "2000"
    else:
        year = year_prefix + date_components[2]
    return f"{day}/{month}/{year}"
format_date_udf = udf(lambda z, year_prefix: get_formatted_date(z, year_prefix), typ.StringType())