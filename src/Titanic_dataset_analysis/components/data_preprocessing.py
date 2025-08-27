import os
import pandas as pd
from Titanic_dataset_analysis import logger
from Titanic_dataset_analysis.utils.common import get_size
from Titanic_dataset_analysis import constants as const
from Titanic_dataset_analysis.entity.config_entity import DataPreprocessingConfig
from pathlib import Path

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config


    
    def read_file(self):
        if not os.path.exists(self.config.input_data_file):
            logger.info(f"File download failed in previous step! Please check the location mentioned : {self.config.input_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.input_data_file))}")  

        self.df = pd.read_csv(self.config.input_data_file)
        logger.info(f"Input file read from {self.config.input_data_file}")
    
    def preprocess_and_save_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        # Basic cleaning / feature engineering
        mean_age = self.df['Age'].mean()
        const.MEAN_AGE = mean_age
        mode_embarked = self.df['Embarked'].mode().iloc[0] if 'Embarked' in self.df.columns else 'S'
        const.MODE_EMBARKED = mode_embarked
        self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce').fillna(mean_age)
        if 'Fare' in self.df.columns:
            self.df['Fare'] = pd.to_numeric(self.df['Fare'], errors='coerce').fillna(0.0)
        self.df['SibSp'] = pd.to_numeric(self.df.get('SibSp', 0), errors='coerce').fillna(0).astype(int)
        self.df['Parch'] = pd.to_numeric(self.df.get('Parch', 0), errors='coerce').fillna(0).astype(int)
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        if 'Embarked' in self.df.columns:
            self.df['Embarked'] = self.df['Embarked'].fillna(mode_embarked)
        
        os.makedirs(self.config.root_dir, exist_ok=True)
        self.df.to_csv(self.config.local_data_file, index=False)
        logger.info(f"Data Preprocessing done successfully.")