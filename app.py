from flask import Flask, request, jsonify, render_template
import os
import json
import mlflow
from flask_cors import CORS, cross_origin
from Titanic_dataset_analysis import logger
import mlflow.spark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pathlib import Path
from Titanic_dataset_analysis.utils.common import read_yaml, load_json
import logging
import sys
logging.getLogger("py4j").setLevel(logging.ERROR)
# logging.getLogger("clientserver").setLevel(logging.ERROR)
python_exec = sys.executable   # path to current python

os.environ["PYSPARK_PYTHON"] = python_exec
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
spark = SparkSession.builder.appName("Flask_app").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
# Define schema consistent with training
titanic_schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True)
])
mlflow_params = load_json(Path("artifacts/mlflow_model_management/model_info.json"))
model_params = read_yaml(Path("artifacts/data_preprocessing/params.yaml"))
mlflow.set_tracking_uri("http://localhost:5000")
model_uri = mlflow_params['model_uri_prod']
model = mlflow.spark.load_model(model_uri)
app = Flask(__name__)
CORS(app)




@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/status", methods=["GET"])
@cross_origin()
def status():
    # print("Status check received.")
    return jsonify({"status": "Model server is running."})


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"

def run_prediction(data):
    """Shared function to handle preprocessing + prediction"""
    pdf = pd.DataFrame(data)
    pdf = pdf.replace({"": None})

    # Ensure numeric columns are floats/ints
    numeric_cols = ["Age", "SibSp", "Parch", "Fare", "Pclass", "PassengerId"]
    for colname in numeric_cols:
        pdf[colname] = pd.to_numeric(pdf[colname], errors="coerce")

    sdf = spark.createDataFrame(pdf, schema=titanic_schema)

    # Drop unused columns
    for drop_col in ["Cabin", "Name", "Ticket"]:
        if drop_col in sdf.columns:
            sdf = sdf.drop(drop_col)

    # Replace NaN with defaults
    for c in ["Age", "Fare", "Pclass", "SibSp", "Parch", "PassengerId"]:
        if c == "Age":
            sdf = sdf.withColumn(c, when(isnan(col(c)) | col(c).isNull(), model_params['MEAN_AGE']).otherwise(col(c)))
        else:
            sdf = sdf.withColumn(c, when(isnan(col(c)) | col(c).isNull(), 0).otherwise(col(c)))

    # Embarked → Fill with training mode
    sdf = sdf.withColumn("Embarked", when(col("Embarked").isNull(), model_params['MODE_EMBARKED']).otherwise(col("Embarked")))

    # Engineered features
    if "FamilySize" not in sdf.columns:
        sdf = sdf.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    if "IsAlone" not in sdf.columns:
        sdf = sdf.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))

    # Run prediction
    predictions = model.transform(sdf)
    pdf_preds = predictions.select("PassengerId", "prediction", "probability").toPandas()

    # Fix DenseVector serialization
    def vec_to_list(x):
        if isinstance(x, DenseVector):
            return x.toArray().tolist()
        return x

    pdf_preds["probability"] = pdf_preds["probability"].apply(vec_to_list)

    return pdf_preds.to_dict(orient="records")

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    try:
        if request.method == "GET":
            # Just render input form
            return render_template('index.html')

        # Check if request is JSON (API call)
        if request.is_json:
            data = request.get_json(force=True)["data"]
            logger.info(f"Data: {data}")
            output = run_prediction(data)
            logger.info(f"Predictions: {output}")
            return jsonify(output)

        # Otherwise assume form submission (UI call)
        json_input = request.form["json_input"]
        data = json.loads(json_input)["data"]
        logger.info(f"Data: {data}")
        output = run_prediction(data)
        logger.info(f"Predictions: {output}")
        return render_template('index.html', output=output)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)})
        else:
            return render_template('index.html', output={"error": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True) #local host