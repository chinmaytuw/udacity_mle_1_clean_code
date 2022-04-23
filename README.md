# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of this project is to implement a clean python package that predicts customer churn.
This entails following coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).



## Files and data description
The directory is set up in the following structure:
.
├── Guide.ipynb          		# Getting started and troubleshooting tips
├── churn_notebook.ipynb 		# Contains the code to be refactored
├── churn_library.py     		# Master functions to run import, perfrom eda, train models, report 
├── churn_script_logging_and_tests.py 	# Testing and logging script
├── README.md            
├── data                 
│   └── bank_data.csv	# data
├── images                
│   ├── eda 				# EDA results
│   └── results 			# classification results
├── logs                 
	├── churn_library.log 		# logs
└── models               	
	├── logistic_model.pkl 		# Logistic regression model pkl
	├── rf_model.pkl 		# RandomForest  model pkl


## Running Files
Below are the steps to run the files:

1. Clone and download the repo to your local machine.
2. Create and activate a new virtual enviornment.
3. Install all the requirements using:
	`python -m pip install -r requirements_py3.8.txt`
4. You can then run `churn_library.py` using the command:
	`python churn_library.py`
5. Next you can run the `churn_script_logging_and_tests.py` using the command:
	`python churn_script_logging_and_tests.py`
6. Lastly, check the log file: `churn_library.log` to see if there were any errors.



