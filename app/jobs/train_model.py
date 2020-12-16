from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassMetrics
from shared.utils import check_numerical_dtype


def _read_data(spark, config, phase):
    return spark.read.parquet(config['processed_data_dir'] + f"{phase}.parquet")


def train_test_split(df, config)
    fractions = df.select("DEFAULT").distinct() \
                  .withColumn("fraction", lit(1-config['test_size'])).rdd.collectAsMap()
    train_df = df.stat.sampleBy(config['target_col'], fractions, config['seed'])
    test_df = df.subtract(train_df)
    return train_df, test_df


def _get_string_indexers(df, config):
    string_indexers = [StringIndexer(inputCol=column, outputCol=column+"_INDEX") \
                        for column in config['str_cat_cols']]
    return string_indexers


def _get_missing_value_imputers(df, config):
    impute_dict = config.get("impute_numerical_cols", None)
    if impute_dict == {} or impute_dict is None:
        return df
    imputers = []
    for column, val in impute_dict.items():
        check_numerical_dtype(df, column)
        imputer = sf.Imputer(inputCols=[column], 
                             outputCols=[column],
                             strategy=val)
        imputers.append(imputer)
    return imputers


def _get_one_hot_encoders(config):
    cat_cols = [f'{column}_INDEX' for column in config['str_cat_cols']]]
    cat_cols += config['cat_cols']
    ohe_encoders = []
    for column in cat_cols:
        ohe_encoders = [OneHotEncoder(inputCol=column,
                                      outputCol="{0}_ENCODED".format(column)) \
                                      for column in cat_cols]
    return ohe_encoders


def _get_vector_assembler(df, config):
    vars_exclude = [config['target_col']] + config['cat_cols'] + config['str_cat_cols']
    assembler_inputs = [c for c in df.columns if c not in vars_exclude]
    vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    return vec_assembler


def _evaluate_results(config, predictions):
    evaluator = BinaryClassificationEvaluator(labelCol=config['target_col'])
    print(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderROC'}))
    print(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderPR'}))
    preds_and_labels = predictions.select(['prediction', config['target_col']]) \ 
                                  .withColumn('label', col(config['target_col']) \
                                  .cast(typ.FloatType())).orderBy('prediction')
    #select only prediction and label columns
    preds_and_labels = preds_and_labels.select(['prediction', 'label'])
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    print(metrics.confusionMatrix().toArray())


def run_job(spark, config, phase):
    """ Runs Data Preparation job"""
    phase = 'train' if "DEFAULT" in raw_df.columns else 'test'
    df = _read_data(spark, config, phase)
    train_df, test_df = train_test_split(df, config)
    # Stratified split:
    string_indexers = _get_string_indexers(config)
    imputers = _get_missing_value_imputers(train_df, config)
    ohe_encoders = _get_one_hot_encoders(config)
    vec_assembler = _get_vector_assembler(train_df, config)
    gbt_clf = GBTClassifier(**config['model_hyperparams'], labelCol='DEFAULT')
    stages = string_indexers + imputers + ohe_encoders + [vec_assembler] + [gbt_clf]
    pipeline = Pipeline(stages=stages).fit(train_df)
    print(f'Saving model to {config['model_path']}')
    pipeline.write().overwrite().save(model_path)
    print(f'Model saved...')
    model = PipelineModel.load(config['model_path'])
    predictions_df = model.transform(test_df)
    _evaluate_results(config, predictions_df)