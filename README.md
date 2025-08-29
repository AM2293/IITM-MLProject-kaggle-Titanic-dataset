# IITM-MLProject-kaggle-Titanic-dataset

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/AM2293/IITM-MLProject-kaggle-Titanic-dataset.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n titanic_env python=3.8 -y
```

```bash
conda activate titanic_env
```

### STEP 02- Change the directory

```bash
cd IITM-MLProject-kaggle-Titanic-dataset
```

### STEP 03- Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 04- From same location in different terminal start MLFlow UI
#### Applicable only for this step

```bash
conda activate titanic_env
mlflow ui
```

### STEP 05- Run python commands or DVC commands

```bash
python main.py
```

```bash
dvc init
dvc repro
dvc dag
```

### STEP 06- RUN application

```bash
python app.py
```

### STEP 07- Open below link in browser
```bash
http://127.0.0.1:5001/predict
```


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