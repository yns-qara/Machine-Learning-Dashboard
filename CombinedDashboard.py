from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import components.HtmlUtils as ut
import plotly.express as px
from sklearn.metrics import confusion_matrix


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


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of estimators", 1, 100)
        max_depth = st.sidebar.slider("Max depth", 1, 100)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))
        params["C"] = C
        params["kernel"] = kernel
    else:
        # penalty = st.sidebar.selectbox("Penalty", ("l1", "l2", "elasticnet"))
        penalty = st.sidebar.selectbox("Penalty", ("l2", "l2"))
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["penalty"] = penalty
        params["C"] = C
    return params


def get_classifier(clf_name, params):
    clf = None
    if clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     random_state=1234)
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], kernel=params["kernel"])
    else:
        clf = LogisticRegression(C=params["C"], penalty=params["penalty"])

    return clf


def app():
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        st.sidebar.title('Data Preprocessing')
        missing_values_threshold = st.sidebar.slider("Missing values threshold", min_value=0.0, max_value=1.0,
                                                     step=0.05, value=0.25)
        target_variable = st.sidebar.text_input("Target variable")
        test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        random_state = st.sidebar.number_input("Random state", value=25)

        data = load_data(file)
        cleaning_options = ['handle missing values', 'encode categorical variables', 'scale numerical variables',
                            'split data']

        selected_option = st.multiselect('Select option:', cleaning_options)

        if 'handle missing values' in selected_option:
            data = handle_missing_values(data, missing_values_threshold)
            # ut.centered_text("Data after handling missing values", 45)
            # st.table(data.head())
        if 'encode categorical variables' in selected_option:
            data = encode_categorical_variables(data)
            # ut.centered_text("Data after encoding categorical variables", 45)
            # st.table(data.head())
        if 'scale numerical variables' in selected_option:
            data = scale_numeric_variables(data)
            # ut.centered_text("Data after scaling numeric variables", 45)
            # st.table(data.head())
        lines_control = st.slider('select number of rows to show : ', 1, len(data), 10)
        st.download_button(
            "Download new data",
            data.to_csv(index=True).encode('utf-8'),
            "new_data.csv",
            "text/csv",
            key='data_download-csv'
        )
        st.write('new data:')
        st.dataframe(data.head(lines_control))
        if 'split data' in cleaning_options:
            if target_variable != "":
                X_train, X_test, y_train, y_test = split_data(data, target_variable, test_size, random_state)
                train_dataset = X_train.to_csv(index=True).encode('utf-8')
                test_dataset = X_test.to_csv(index=True).encode('utf-8')
                train_variable = y_train.to_csv(index=True).encode('utf-8')
                test_variable = y_test.to_csv(index=True).encode('utf-8')

                st.write("Training Data")
                st.table(X_train.head())
                st.write("Testing Data")
                st.table(X_test.head())

                st.write("X_train : ")
                st.download_button(
                    "Download",
                    train_dataset,
                    "X_train.csv",
                    "text/csv",
                    key='download1-csv'
                )
                st.write("X_test : ")
                st.download_button(
                    "Download",
                    test_dataset,
                    "X_test.csv",
                    "text/csv",
                    key='download2-csv'
                )

                st.write("y_train : ")

                st.download_button(
                    "Download",
                    train_variable,
                    "Y_train.csv",
                    "text/csv",
                    key='download3-csv'
                )
                st.write("y_test : ")
                st.download_button(
                    "Download",
                    test_variable,
                    "Y_test.csv",
                    "text/csv",
                    key='download4-csv'
                )
            elif target_variable not in data.columns and target_variable != '':
                st.sidebar.warning("no such column name")

        st.sidebar.title("Training")

        ml_problem = st.sidebar.radio(
            "select the machine learning problem",
            ('None', 'Classification', 'Regression', 'Forecasting'))

        if ml_problem == 'None':
            pass
        elif ml_problem == 'Classification':

            st.sidebar.subheader("Select Classifier")
            problem_type = st.sidebar.selectbox("Classifier", ("Random Forest", "SVM", "Logistic Regression"))

            params = add_parameter_ui(problem_type)
            classifier = get_classifier(problem_type, params)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.title('TRAINING')

            st.write(f"Accuracy = {acc}")
            st.progress(acc)
            st.write('confusion matrix : ')
            # cm = confusion_matrix(y_test, y_pred)
            # fig = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"))
            # st.plotly_chart(fig)

            st.write("Download the trained model")

            model_download = st.button("Download")
            if model_download:
                filename = f"{classifier}.pkl"
                with open(filename, 'wb') as file:
                    pickle.dump(classifier, file)


        elif ml_problem == 'Regression':

            pass

        elif ml_problem == 'Forecasting':
            pass





    else:
        ut.centered_title("Machine Learning Dashboard")


if __name__ == "__main__":
    app()
