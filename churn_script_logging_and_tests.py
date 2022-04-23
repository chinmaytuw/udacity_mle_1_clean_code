'''
The is a library of functions to test churn_library.py

Author:Chinmay
Date: 4/19/22
'''

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        imported_df = cl.import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert imported_df.shape[0] > 0
        assert imported_df.shape[1] > 0
        logging.info("SUCCESS: DF shape checks out")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    # Read data
    imported_df = cl.import_data("./data/bank_data.csv")

    try:
        cl.perform_eda(imported_df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error("Column {} not found".format(err.args[0]))

    try:
        assert os.path.isfile("./images/eda/churn_histogram.png") is True
        logging.info("SUCCESS: churn_histogram found")
    except AssertionError:
        logging.error("ERROR: churn_histogram not found on disk")

    try:
        assert os.path.isfile(
            "./images/eda/customer_age_histogram.png") is True
        logging.info("SUCCESS: customer_age_histogram found")
    except AssertionError:
        logging.error("ERROR: customer_age_histogram not found on disk")

    try:
        assert os.path.isfile("./images/eda/marital_status.png") is True
        logging.info("SUCCESS: marital_status found")
    except AssertionError:
        logging.error("ERROR: marital_status not found on disk")

    try:
        assert os.path.isfile("./images/eda/total_transaction_kde.png") is True
        logging.info("SUCCESS: total_transaction_kde found")
    except AssertionError:
        logging.error("ERROR: total_transaction_kde not found on disk")

    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info("SUCCESS: heatmap found")
    except AssertionError:
        logging.error("ERROR: heatmap not found on disk")


def test_encoder_helper():
    '''
    test encoder helper
    '''
    # Read data
    imported_df = cl.import_data("./data/bank_data.csv")

    # List of categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    ### SCENARIO 1: Ideal case where we pass cal_columns ##########
    try:
        encoded_df = cl.encoder_helper(
            imported_df, cat_columns)  # Pass df and cat_columns
        # assert that both dataframes look different
        assert encoded_df.equals(imported_df) is False
        logging.info("SUCCESS Scenario 1: Dataframe encoded with cat_columns")
    except AssertionError:
        logging.error("ERROR: Unable to encode for cat_columns")

    ### SCENARIO 2: Pass nothing in place of cal_columns ##########
    try:
        encoded_df = cl.encoder_helper(imported_df, [])
        # assert that both dataframes look same
        assert encoded_df.equals(imported_df) is True
        logging.info("SUCCESS Scenario 2: Dataframe encoded with []")
    except AssertionError:
        logging.error("ERROR: Unable to encode for []")

    ### SCENARIO 3: Pass cat_columns with reponse: '_test' ##########
    try:
        encoded_df = cl.encoder_helper(imported_df, cat_columns, 'test')
        test_cols = [col for col in encoded_df.columns if 'test' in col]
        # assert that all cat_columns are encoded
        assert len(test_cols) == len(cat_columns)
        logging.info(
            "SUCCESS Scenario 3: Dataframe encoded with cat_columns and respose: _test")
    except AssertionError:
        logging.error("ERROR: Unable to encode for []")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    # Read data
    imported_df = cl.import_data("./data/bank_data.csv")

    # List of categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # encode df
    encoded_df = cl.encoder_helper(imported_df, cat_columns)

    # SCENARIO 1: Pass no argument to perform_feature_engineering causing
    # failure ####
    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(encoded_df)
        logging.info("SUCCESS: able to perform train test split")
    except TypeError:
        logging.error(
            "ERROR: Unable to perform train test split since the only argument is missing")

    ### SCENARIO 2: Ideal scenario where we pass encoded_df ####
    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            encoded_df)
        # Check number of column in x_test and x_train are same
        assert x_train.shape[1] == x_test.shape[1]

        # Check number of rows in x_train and y_train are same
        assert x_train.shape[0] == y_train.shape[0]

        # Check number of rows in x_test and y_test are same
        assert x_test.shape[0] == y_test.shape[0]

        logging.info("SUCCESS: able to perform train test split")

    except AssertionError:
        logging.error("ERROR: Training and testing data shapes don't match")


def test_train_models():
    '''
    test train_models
    '''
    # Read data
    imported_df = cl.import_data("./data/bank_data.csv")

    # List of categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # encode df
    encoded_df = cl.encoder_helper(imported_df, cat_columns)

    # Get train test split
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        encoded_df)

    # Train models
    try:
        cl.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('SUCCESS: Models trained, RandomForest model found')
    except AssertionError:
        logging.error(
            "ERROR: Models not trained, RandomForest model not found")

    try:
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info(
            'SUCCESS: Models trained, LogisticRegression  model found')
    except AssertionError:
        logging.error(
            "ERROR: Models not trained, LogisticRegression model not found")

    try:
        assert os.path.isfile("./images/results/roc_curve.png") is True
        logging.info('SUCCESS: ROC Curve found')

    except AssertionError:
        logging.error("ERROR: ROC Curve not found")

    try:
        assert os.path.isfile("./images/results/roc_curve.png") is True
        logging.info('SUCCESS: ROC Curve found')
    except AssertionError:
        logging.error("ERROR: ROC Curve not found")

    try:
        assert os.path.isfile("./images/results/rf_results.png") is True
        logging.info('SUCCESS: RandomForest results found')
    except AssertionError:
        logging.error("ERROR: RandomForest results not found")

    try:
        assert os.path.isfile("./images/results/lr_results.png") is True
        logging.info('SUCCESS: LogisticRegression results found')
    except AssertionError:
        logging.error("ERROR: LogisticRegression results not found")

    try:
        assert os.path.isfile(
            "./images/results/feature_importances.png") is True
        logging.info('SUCCESS: feature_importances found')
    except AssertionError:
        logging.error("ERROR: feature_importances not found")


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
