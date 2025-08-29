import os
import json
import mlflow
from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.ml import PipelineModel
from Titanic_dataset_analysis import logger
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from Titanic_dataset_analysis import constants as const
from Titanic_dataset_analysis.entity.config_entity import MLFlowModelManagementConfig

class MLFlowModelManagement:
    def __init__(self, config: MLFlowModelManagementConfig):
        self.config = config


    
    def load_model(self):
        if not os.path.exists(self.config.input_model_folder):
            logger.info(f"Model download failed in previous step! Please check the location mentioned : {self.config.input_data_file}")
        else:
            logger.info(f"Model already exists at: {Path(self.config.input_model_folder)}")  

        self.model = PipelineModel.load(self.config.input_model_folder)
        logger.info(f"Loaded model from {self.config.input_model_folder}")
    
    def mlflow_model_tracking(self):
        """
        self.model is having the current model.
        create the spark session and read the preprocessed splitted test_data for parameter measurement.
        Log all those parameters in the mlflow experiment tracking and model registry.
        It saves the mlflow parameters at model_info.json.
        Function returns None
        """
        # -------------------------------
        # MLflow Tracking
        # -------------------------------
        spark = SparkSession.builder.appName(self.config.params_sparkSessionTitle).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        
        df = spark.read.csv(str(self.config.test_data_file), header=True, inferSchema=True)
        logger.info(f"Experiment Name: {self.config.params_experiment_name}")
        experiment_name = self.config.params_experiment_name
        logger.info(f"Setting Tracking URI to : {self.config.params_mlflow_uri}")
        mlflow.set_tracking_uri(self.config.params_mlflow_uri)
        os.environ["MLFLOW_ARTIFACT_URI"] = f"file:./mlruns"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment ID: {experiment_id}")
        with mlflow.start_run(run_name=self.config.params_mlflow_run_name) as run:
            best_lr = self.model.stages[-1]
            
            logger.info(f"regparam: {best_lr.getOrDefault('regParam')}")
            logger.info(f"ElasticNetParam: {best_lr.getOrDefault('elasticNetParam')}")
            mlflow.log_param("regParam", best_lr.getOrDefault("regParam"))
            mlflow.log_param("elasticNetParam", best_lr.getOrDefault("elasticNetParam"))
    
            predictions = self.model.transform(df)
            
            # Evaluator
            evaluator = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
            # Metrics
            auc_test = evaluator.evaluate(predictions)
            mlflow.log_metric("AUC", auc_test)
            logger.info(f"AUC: {auc_test}")

            accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
            precision = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
            recall = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
            f1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1").evaluate(predictions)

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1", f1)
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1: {f1}")
            
            
            mlflow.spark.log_model(self.model,
                           artifact_path=self.config.params_mlflow_run_name,
                           registered_model_name=self.config.params_experiment_name)

            const.RUN_ID = run.info.run_id
            const.EXPERIMENT_ID = run.info.experiment_id

            logger.info(f"Run ID from constants: {run.info.run_id}")
            logger.info(f"Experiment ID from constants: {run.info.experiment_id}")

            metadata = {
                "experiment_id": run.info.experiment_id,
                "run_id": run.info.run_id,
                "model_uri": f"runs:/{run.info.run_id}/{self.config.params_mlflow_run_name}",
                "model_uri_prod": f"models:/{experiment_name}/Production"
            }
            with open("artifacts/mlflow_model_management/model_info.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved model metadata: {metadata}")
                
        os.makedirs(self.config.root_dir, exist_ok=True)
        logger.info(f"MLFlow Model Tracking done successfully.")
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(self.config.params_experiment_name, stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0].version
            client.transition_model_version_stage(
                name=self.config.params_experiment_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"Model '{self.config.params_experiment_name}' version {latest_version} moved to Production!")
        spark.stop()