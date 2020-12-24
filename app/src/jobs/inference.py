import time
from pyspark.ml import PipelineModel


def _read_data(spark, config, mode):
    if mode == "local":
        return spark.read.parquet(config['processed_data_dir'] + "predict.parquet")
    else:
        return spark.read.parquet(config['s3_processed_data_dir'].format(s3_bucket) + "predict.parquet")


def run_job(spark, config, mode, s3_bucket=None, phase=None):
    """ Runs Model Training job"""
    df = _read_data(spark, config, mode)
    pipeline = Pipeline(stages=stages).fit(train_df)
    if mode == "local":
        model_path = config['model_path']
    else:
        model_path = config['s3_model_path'].format(s3_bucket)
    print('Loading model from {}'.format(model_path))
    model = PipelineModel.load(model_path)
    predictions_df = model.transform(df)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    predictions_df.write.parquet(config['predictions_dir'].format(s3_bucket) + \
                                 f"{}_prediction.parquet".format(timestr), 
                                 mode='overwrite')