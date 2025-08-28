from Titanic_dataset_analysis.config.configuration import ConfigurationManager
from Titanic_dataset_analysis.components.mlflow_model_management import MLFlowModelManagement
from Titanic_dataset_analysis import logger


STAGE_NAME = "MLFlow Model Tracking stage"

class MLFlowModelManagementPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        mlflow_model_tracking_config = config.mlflow_model_management_config()
        mlflow_model_tracking_config = MLFlowModelManagement(config=mlflow_model_tracking_config)
        mlflow_model_tracking_config.load_model()
        mlflow_model_tracking_config.mlflow_model_tracking()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = MLFlowModelManagementPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e