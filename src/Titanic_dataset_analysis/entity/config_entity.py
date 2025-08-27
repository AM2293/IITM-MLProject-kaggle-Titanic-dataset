from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    input_data_file: Path
    local_data_file: Path
    

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    input_data_file: Path
    model_file: Path
    params_splitratio: list
    params_seed: int
    params_regParam: list
    params_elasticNetParam: list
    params_number_of_folds: int
    params_sparkSessionTitle: str