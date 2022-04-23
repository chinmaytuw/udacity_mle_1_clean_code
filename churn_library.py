'''
The is a library of functions to find customers who are likely to churn.

Author:Chinmay
Date: 4/19/22
'''

####################################   Import libraries #################
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# os.environ['QT_QPA_PLATFORM']='offscreen'

####################################   Set static variables #############
# List of categorical columns
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


# List of columns to keep
keep_cols = ['Customer_Age',
             'Dependent_count',
             'Months_on_book',
             'Total_Relationship_Count',
             'Months_Inactive_12_mon',
             'Contacts_Count_12_mon',
             'Credit_Limit',
             'Total_Revolving_Bal',
             'Avg_Open_To_Buy',
             'Total_Amt_Chng_Q4_Q1',
             'Total_Trans_Amt',
             'Total_Trans_Ct',
             'Total_Ct_Chng_Q4_Q1',
             'Avg_Utilization_Ratio',
             'Gender_Churn',
             'Education_Level_Churn',
             'Marital_Status_Churn',
             'Income_Category_Churn',
             'Card_Category_Churn']

####################################   Define Functions #################


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # Read file
    df = pd.read_csv(pth)

    # Add churn flag column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        # Plot histogram for churn
        plt.figure(figsize=(20, 10))
        df['Churn'].hist()
        plt.title('Histogram of Chrun')
        plt.savefig('images/eda/churn_histogram.png')
        plt.clf()
    except KeyError:
        pass

    try:
        # Plot histogram for Customer's age
        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.title('Histogram of Customer\'s Age')
        plt.savefig('images/eda/customer_age_histogram.png')
        plt.clf()
    except KeyError:
        pass

    try:
        # Plot normalized count of marital status
        plt.figure(figsize=(20, 10))
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.title('Normalized count of marital status')
        plt.savefig('images/eda/marital_status.png')
        plt.clf()
    except KeyError:
        pass

    try:
        # Plot to show distributions of 'Total_Trans_Ct' with a smooth curve
        # using a kernel density estimate
        plt.figure(figsize=(20, 10))
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Desnity plot for total transactions')
        plt.savefig('images/eda/total_transaction_kde.png')
        plt.clf()
    except KeyError:
        pass

    try:
        # Heatmap to show correlation b/w different numerical variables
        plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title('Heatmap of numerical variables')
        plt.savefig('images/eda/heatmap.png')
        plt.clf()
    except KeyError:
        pass


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                [optional argument that could be used for naming variables
                or index y column]

    output:
            df: pandas dataframe with new columns for propotion
    '''
    encoded_df = df.copy()

    for col in category_lst:
        encoded_df[col + '_' + response] = encoded_df[col].map(
            encoded_df.groupby(col)['Churn'].mean())

    return encoded_df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Split features and labels
    y = df['Churn']
    X = df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    #################### RandomForest ######################
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')
    plt.clf()

    #################### LogisticRegression  ######################
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/lr_results.png')
    plt.clf()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    feature_importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importances for RandomForestClassifier")
    plt.ylabel('Importance score')
    # Add bars
    plt.bar(range(x_data.shape[1]), feature_importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')
    plt.clf()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    #################### RandomForest ######################
    # Instantiate RandomForest model
    rfc = RandomForestClassifier(random_state=42)

    # Create parameter grid for Random forest's Gridsearch
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Instantiate Grid search with 5 fold cross validation
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Fit grid search model to training data
    cv_rfc.fit(x_train, y_train)

    # Get prediction on both training and test sets via the best estimator
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    #################### Logistic Regression ######################
    # Instantiate Logistic regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Fit model
    lrc.fit(x_train, y_train)

    # Get prediction on both training and test sets
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Save model
    joblib.dump(lrc, './models/logistic_model.pkl')

    #################### Plot joint ROC curves #######################
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test,
                   ax=axis, alpha=0.8)

    plt.savefig(fname='./images/results/roc_curve.png')
    plt.clf()

    #################### Generate Report and Importance #######################
    # Generate classification report
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            x_data=x_test,
                            output_pth='./images/results/')


if __name__ == "__main__":
    print('Starting Run')

    df = import_data('./data/bank_data.csv')

    perform_eda(df)

    # perform encoding
    df = encoder_helper(df, cat_columns)

    # Feature engg
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(df)
    print(X_TRAIN.shape)

    # training
    train_models(x_train=X_TRAIN,
                 x_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST)
