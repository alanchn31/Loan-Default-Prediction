## Loan Default Prediction

  
## Problem Statement
---

* Financial institutions incur significant losses due to the default of vehicle loans.

* <ins>Goal:</ins> To build a machine learning model that can accurately predict borrowers that are likely to default on first EMI (Equated Monthly Instalments)

## Set-up
---
* Run the following commands to submit spark job (in app dir):  
    1. Run ```make build``` to move all dependencies for spark job to dist/ folder
    2. Run ```spark-submit --py-files dist/jobs.zip --files config.json main.py --job prepare_data``` to submit spark job

* Running pytest:  
    * Run ```make test``` (in app dir)

## WIP
---

<ins>Completed</ins>
* Prototyping Model Training Methodology

<ins>WIP</ins>
* Structuring code for building pipeline for model training and deployment