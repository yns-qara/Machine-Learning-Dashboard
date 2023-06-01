
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import components.HtmlUtils as ut

@st.cache
def load_data(file):
    """ load data as csv from the os, and skip the lines where the number of columns is greater than number of
    parameters """
    data = pd.read_csv(file, error_bad_lines=False)

    return data


def handle_missing_values(data, threshold):
    missing_values = data.isnull().sum() / len(data)
    missing_values = missing_values[missing_values >= threshold].index
    data = data.drop(missing_values, axis=1)
    data = data.fillna(data.mean())
    return data


def encode_categorical_variables(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].astype('category')
        data[col] = data[col].cat.codes
    return data


def scale_numeric_variables(data):
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data


def split_data(data, target_variable, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target_variable, axis=1), data[target_variable],
                                                        test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def app():
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    missing_values_threshold = st.sidebar.slider("Missing values threshold", min_value=0.0, max_value=1.0, step=0.05,
                                                 value=0.25)
    target_variable = st.sidebar.text_input("Target variable")
    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
    random_state = st.sidebar.number_input("Random state", value=25)


    if file is not None:
        data = load_data(file)
        ut.centered_text("Original Data", 45)
        st.table(data.head(20))

        data = handle_missing_values(data, missing_values_threshold)
        ut.centered_text("Data after handling missing values", 45)
        st.table(data.head())

        data = encode_categorical_variables(data)
        ut.centered_text("Data after encoding categorical variables", 45)
        st.table(data.head())

        data = scale_numeric_variables(data)
        ut.centered_text("Data after scaling numeric variables", 45)
        st.table(data.head())

        # add a condition when the target variable is wrong
        if target_variable != "":
            X_train, X_test, y_train, y_test = split_data(data, target_variable, test_size, random_state)
            train_dataset = X_train.to_csv(index=True).encode('utf-8')
            test_dataset = X_test.to_csv(index=True).encode('utf-8')
            train_variable = y_train.to_csv(index=True).encode('utf-8')
            test_variable = y_test.to_csv(index=True).encode('utf-8')


            st.write("Training Data")
            st.table(X_train.head())
            st.download_button(
                "Download X_train",
                train_dataset,
                "X_train.csv",
                "text/csv",
                key='download1-csv'
            )
            st.write("Testing Data")
            st.table(X_test.head())
            st.download_button(
                "Download X_test",
                test_dataset,
                "X_test.csv",
                "text/csv",
                key='download2-csv'
            )

            st.write("additional downloads")

            st.download_button(
                "Download y_train",
                train_variable,
                "Y_train.csv",
                "text/csv",
                key='download3-csv'
            )
            st.download_button(
                "Download y_test",
                test_variable,
                "Y_test.csv",
                "text/csv",
                key='download4-csv'
            )

    else :
        ut.centered_title("Machine Learning Dashboard")



# la fonction main
if __name__ == "__main__":
    app()
