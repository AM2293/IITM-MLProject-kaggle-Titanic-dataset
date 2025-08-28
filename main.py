from Titanic_dataset_analysis import logger
from Titanic_dataset_analysis import constants as c
from Titanic_dataset_analysis.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Titanic_dataset_analysis.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from Titanic_dataset_analysis.pipeline.stage_03_model_training import ModelTrainingPipeline
from Titanic_dataset_analysis.pipeline.stage_04_mlflow_model_management import MLFlowModelManagementPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    dataingestion = DataIngestionTrainingPipeline()
    dataingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Preprocessing Stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   datapreprocessing = DataPreprocessingPipeline()
   datapreprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
    
STAGE_NAME = "Model Training stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    modeltraining = ModelTrainingPipeline()
    modeltraining.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MLFlow Model Tracking stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    MLFlowModelManagement = MLFlowModelManagementPipeline()
    MLFlowModelManagement.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e