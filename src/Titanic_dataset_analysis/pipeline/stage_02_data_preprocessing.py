from Titanic_dataset_analysis.config.configuration import ConfigurationManager
from Titanic_dataset_analysis.components.data_preprocessing import DataPreprocessing
from Titanic_dataset_analysis import logger


STAGE_NAME = "Data Ingestion stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.read_file()
        data_preprocessing.preprocess_and_save_file()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e