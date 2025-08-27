from Titanic_dataset_analysis.config.configuration import ConfigurationManager
from Titanic_dataset_analysis.components.model_training import ModelTraining
from Titanic_dataset_analysis import logger


STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.read_file()
        model_training.model_training_and_save_file()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e