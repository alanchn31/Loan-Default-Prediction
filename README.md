## Smart Lending Default Prediction

![Screenshot](docs/images/project_icon.PNG)

(Source of icon: https://www.flaticon.com/free-icon/car-loan_2505972)

## How to use this repo?
---
* This repo can act as an example on how to structure pyspark code to train a machine learning model.
* Data transformations are modularized into Transformers (app/transformers) then subsequently composed into a pipeline in our preprocess script (app/jobs/preprocess.py)
* After preprocessing, we fit our Estimators and compose a specialized pipeline that can be fitted on our training data and be used to transform/predict external data. (see app/jobs/train_model.py)
* See (Set-up) on how to set up environment and run scripts.


## Domain Background
---

* Financial institutions incur significant losses due to the default of vehicle loans every year.

  
## Problem Statement
---

* <ins>Goal:</ins> To accurately pre-empt problematic borrowers who are likely to default in future, our goal is to build a machine learning model that can accurately predict borrowers that are likely to default on first EMI (Equated Monthly Instalments)


## Data
---
* We used a public dataset that can be found on Kaggle: https://www.kaggle.com/avikpaul4u/vehicle-loan-default-prediction.

* Dataset has 50,000 rows with 41 columns

* As data is imbalanced, (about 80% of labels are non-default while 20% are defaults), our main metric of evaluating our trained model will be the AUC_ROC


## Architecture of Model Training Pipeline
---
![Screenshot](docs/images/modelling_architecture.PNG)


## Results
---
* AUC_ROC: 0.625

## Set-up
---
Note that this project requires spark to be installed on a local system. 

* Run the following command for initial set-up of virtual environment:
    * Run ```make setup``` on home dir (outside app dir)

* Run the following commands to submit spark job (in app dir):  
    1. Run ```make build``` to move all dependencies for spark job to dist/ folder
    2. Run ```make preprocess_train``` to submit spark job to preprocess training data
    3. Run ```make train_model``` to submit spark job to train our model on preprocessed training data

* Running pytest:  
    * Run ```make test``` (in app dir)