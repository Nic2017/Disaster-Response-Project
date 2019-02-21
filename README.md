# Disaster Response Project

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Installation
* Python 3.*
* Basic Python Packages: sys, pandas, string, json, numpy, collections, pickle
* Machine Learning and NLP Packages: sklearn, nltk
* SQLite Database Package: sqlalchemy
* Web App and Data Visualization Packages: flask, plotly

## File Structure:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- disaster_model.pkl  # saved model

- notebooks
|- ETL Pipeline Preparation.ipynb # etl exploration
|- ML Pipeline Preparation.ipynb # ML exploration

- README.md
``` 

## Authors, License and Acknowledgements
### Author: 
[Nicolas Guan](https://github.com/Nic2017)

### License: 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Acknowledgements: 
Thanks to [Udacity](https://www.udacity.com/) for providing such a thorough and comprehensive project combining ETL Pipeline+ML Pipeline+Web App development. 
Also thanks to [Figure Eight](https://www.figure-eight.com/) for providing training dataset for the model.
