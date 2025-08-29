# IITM-MLProject-kaggle-Titanic-dataset

## Steps to run the project
1. git clone <Repo link>  ## Can be HTTP/SSH
2. install conda
3. conda create --name <env name> python=3.8 -y
4. conda activate <env name>
5. change the directory to cd IITM-MLProject-kaggle-Titanic-dataset
6. pip install -r requirements.txt
7. from same location in different terminal, start MLFlow UI for experimant visualisation
   mlflow ui
8. python main.py / dvc repro  --> For Running the pipeline
9. python app.py  --> For running the flask app to check for predictions.

## Workflows to update the folder in modular format
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml