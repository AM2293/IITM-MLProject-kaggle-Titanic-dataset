# from Titanic_dataset_analysis import constants as c
from Titanic_dataset_analysis.constants import *
# CONFIG_FILE_PATH = Path("config/config.yaml")
# PARAMS_FILE_PATH = Path("params.yaml")
from Titanic_dataset_analysis.utils.common import read_yaml, create_directories
from Titanic_dataset_analysis.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            input_data_file=config.input_data_file,
            local_data_file=config.local_data_file
        )

        return data_preprocessing_config