import pickle
import dataCleaning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def get_dataFrame() -> pd.DataFrame:
    # Get the cleaned data
    chronic_indicator_data = dataCleaning.main()
    return chronic_indicator_data

def get_splitData(df):
    
    # Filter the dataset for the specific topic "Cancer"
    df_cancer = df[df['topic'] == 'Cancer']

    # Select relevant columns
    selected_columns = ['locationdesc', 'datavalue', 'stratification1']
    df_cancer = df_cancer[selected_columns].dropna()  # Drop rows with missing values

    # print(",,,,,,,,,,,,,,,,,,,,,,,", df_cancer.shape)
    # Convert categorical variables to numerical
    df_cancer = pd.get_dummies(df_cancer, columns=['locationdesc', 'stratification1'])
    # print("################",df_cancer.shape)

    # Separate features (X) and target variable (y)
    X = df_cancer.drop('datavalue', axis=1)
    y = df_cancer['datavalue']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def model_training(X,Y):

    # Build the decision tree regression model
    dt_model = DecisionTreeRegressor(random_state=42)

    # Train the model
    dt_model.fit(X, Y)

    # Build the random forest regression model
    rf_model = RandomForestRegressor(random_state=42)

    # Train the model
    rf_model.fit(X, Y)

    return dt_model, rf_model
    

def main ():
    chronic_data = get_dataFrame()
    XTrain, XTest, labelsTrain, labelsTest = get_splitData(chronic_data)

    dTree_model, rForest_model = model_training(XTrain, labelsTrain)

    with open ('final_dt_model.pkl', 'wb') as dtfile:
        pickle.dump(dTree_model, dtfile)

    with open ('final_rf_model.pkl', 'wb') as dtfile:
        pickle.dump(rForest_model, dtfile)

if __name__ == '__main__':
    main()