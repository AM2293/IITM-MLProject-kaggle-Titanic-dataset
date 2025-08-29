import os
import pandas as pd
import findspark
from pathlib import Path
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from Titanic_dataset_analysis import logger
from Titanic_dataset_analysis.utils.common import get_size
from Titanic_dataset_analysis.entity.config_entity import ModelTrainingConfig

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config


    
    def read_file(self):
        if not os.path.exists(self.config.input_data_file):
            logger.info(f"File download failed in previous step! Please check the location mentioned : {self.config.input_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.input_data_file))}")

        self.df = pd.read_csv(self.config.input_data_file)
        logger.info(f"Processed Input file read from {self.config.input_data_file}")
    
    def model_training_and_save_file(self):
        """
        self.df has the preprocessed data.
        We create spark session form pipeline of indexers-> encoders-> Vector Assembler-> LogisticRegression
        Check the best params for LR model and then save them.
        root_dir: Path. Here we save the model that is being trained on the processed data.
        Function returns None
        """
        # print(self.config)
        # logger.info(f"Spark session title is: {self.config.params_sparkSessionTitle}")
        spark = SparkSession.builder.appName(self.config.params_sparkSessionTitle).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        
        df = spark.read.csv(str(self.config.input_data_file), header=True, inferSchema=True)
        # logger.info(df.show())
        # Drop Cabin + Name + Ticket (not useful for ML in our setup)
        columns_to_drop = ["Cabin", "Name", "Ticket"]
        # Categorical feature processing
        categorical_cols = ["Sex", "Embarked"]
        indexed_cols = [c + "_indexed" for c in categorical_cols]
        encoded_cols = [c + "_encoded" for c in categorical_cols]

        indexers = [StringIndexer(inputCol=c, outputCol=c + "_indexed", handleInvalid="keep") for c in categorical_cols]
        encoders = [OneHotEncoder(inputCol=ic, outputCol=ec) for ic, ec in zip(indexed_cols, encoded_cols)]

        # Vector Assembler
        feature_columns = [
            "Pclass", "Age", "SibSp", "Parch", "Fare",
            "FamilySize", "IsAlone",
            "Sex_encoded", "Embarked_encoded"
        ]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # Logistic Regression model
        lr = LogisticRegression(featuresCol="features", labelCol="Survived")
        # -------------------------------
        # Pipeline
        # -------------------------------
        pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

        # -------------------------------
        # Train/Test split
        # -------------------------------
        train_data, test_data = df.drop(*columns_to_drop).randomSplit(self.config.params_splitratio, seed=self.config.params_seed)
        test_data.write.mode("overwrite").option("header", True).csv(self.config.test_data_file)
        logger.info(f"Test Data is saved at: {self.config.test_data_file}")

        # -------------------------------
        # Param Grid & CrossValidator
        # -------------------------------
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, self.config.params_regParam) \
            .addGrid(lr.elasticNetParam, self.config.params_elasticNetParam) \
            .build()

        # Evaluator
        evaluator = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

        crossval = CrossValidator(estimator=pipeline,
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator,
                                numFolds=self.config.params_number_of_folds)
        cv_model = crossval.fit(train_data)
        best_lr = cv_model.bestModel.stages[-1]
        logger.info(f"Best model: {best_lr}")
        
        os.makedirs(self.config.root_dir, exist_ok=True)
        model_path =f"{self.config.root_dir}/best_model"
        if os.path.exists(model_path):
            cv_model.bestModel.write().overwrite().save(model_path)
        else:
            cv_model.bestModel.write().save(model_path)
        logger.info(f"Model training done successfully. Model saved at {self.config.root_dir}/best_model")
        
        spark.stop()