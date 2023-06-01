from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import components.HtmlUtils as ut
import plotly.express as px
from sklearn.linear_model import Perceptron, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(layout="wide", page_title="ML Dashboard")





class DataPreprocessing:
    def __init__(self, file):
        self.data = pd.read_csv(file, error_bad_lines=False)

    def handle_missing_values(self, threshold):
        missing_values = self.data.isnull().sum() / len(self.data)
        missing_values = missing_values[missing_values >= threshold].index
        data = self.data.drop(missing_values, axis=1)
        data = data.fillna(data.mean())
        return data

    def encode_categorical_variables(self, data):
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
        return data

    def scale_numeric_variables(self, data, target_variable):
        new_data = data.loc[:, data.columns != target_variable]
        numeric_columns = new_data.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        return data

    def split_data(self, data, target_variable, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(data.drop(target_variable, axis=1), data[target_variable],
                                                            test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


class ModelBuilding:
    def __init__(self, data, target_variable):
        pass

    def add_parameter_ui(self, clf_name):
        params = dict()
        if clf_name == "Logistic Regression":
            params["C"] = st.sidebar.slider('C', 0.01, 100.0, 1.0, 0.01)
            class_weight_options = [None, 'balanced', {'class_0': 1, 'class_1': 10}]
            params["class_weight"] = st.sidebar.selectbox('Class Weight', class_weight_options, index=0)
            params["dual"] = st.sidebar.checkbox('Dual', False)
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            # params["intercept_scaling"] = st.sidebar.slider('Intercept Scaling', 0.1, 10.0, 1, 0.1)
            params["l1_ratio"] = st.sidebar.slider('L1 Ratio', 0.0, 1.0, None, 0.01)
            params["max_iter"] = st.sidebar.slider('Max Iterations', 10, 1000, 100, 10)
            params["multi_class"] = st.sidebar.selectbox('Multi Class', ['ovr', 'multinomial', 'auto'], index=2)
            params["n_jobs"] = st.sidebar.slider('Number of Jobs', -1, 16, None, 1)
            params["penalty"] = st.sidebar.selectbox('Penalty', ['l1', 'l2', 'elasticnet', 'none'], index=1)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["solver"] = st.sidebar.selectbox('Solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                                    index=1)
            params["tol"] = st.sidebar.slider('Tolerance', 0.0001, 0.1, 0.0001, 0.0001)
            params["verbose"] = st.sidebar.slider('Verbose', 0, 10, 0, 1)
            params["warm_start"] = st.sidebar.checkbox('Warm Start', False)
        elif clf_name == "Perceptron":
            params["alpha"] = st.sidebar.slider('Alpha', 0.0001, 1.0, 0.0001, 0.0001)
            class_weight_options = [None, 'balanced']
            params["class_weight"] = st.sidebar.selectbox('Class Weight', class_weight_options, index=0)
            params["early_stopping"] = st.sidebar.checkbox('Early Stopping', False)
            params["eta0"] = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.1, 0.001)
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            params["l1_ratio"] = st.sidebar.slider('L1 Ratio', 0.0, 1.0, 0.15, 0.01)
            params["max_iter"] = st.sidebar.slider('Max Iterations', 100, 10000, 1000, 100)
            params["n_iter_no_change"] = st.sidebar.slider('Iterations No Change', 1, 50, None, 1)
            params["n_jobs"] = st.sidebar.slider('Number of Jobs', -1, 16, None, 1)
            penalty_options = ['l2', 'l1', 'elasticnet']
            params["penalty"] = st.sidebar.selectbox('Penalty', penalty_options, index=0)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["shuffle"] = st.sidebar.checkbox('Shuffle', True)
            params["tol"] = st.sidebar.slider('Tolerance', 0.0001, 0.1, 0.0001, 0.0001)
            params["validation_fraction"] = st.sidebar.slider('Validation Fraction', 0.1, 0.5, 0.1, 0.1)
            params["verbose"] = st.sidebar.slider('Verbose', 0, 10, 0, 1)
            params["warm_start"] = st.sidebar.checkbox('Warm Start', False)
        elif clf_name == "Linear SVC":
            params["C"] = st.sidebar.slider('C', 0.1, 10.0, 1.0, 0.1)
            class_weight_options = [None, 'balanced']
            params["class_weight"] = st.sidebar.selectbox('Class Weight', class_weight_options, index=0)
            params["dual"] = st.sidebar.checkbox('Dual', True)
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            params["intercept_scaling"] = st.sidebar.slider('Intercept Scaling', 0.1, 10.0, 1.0, 0.1)
            loss_options = ['hinge', 'squared_hinge']
            params["loss"] = st.sidebar.selectbox('Loss', loss_options, index=0)
            params["max_iter"] = st.sidebar.slider('Max Iterations', 100, 10000, 1000, 100)
            multi_class_options = ['ovr', 'crammer_singer']
            params["multi_class"] = st.sidebar.selectbox('Multi Class', multi_class_options, index=0)
            penalty_options = ['l2', 'l1']
            params["penalty"] = st.sidebar.selectbox('Penalty', penalty_options, index=0)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["tol"] = st.sidebar.slider('Tolerance', 0.0001, 0.1, 0.0001, 0.0001)
            params["verbose"] = st.sidebar.slider('Verbose', 0, 10, 0, 1)
        elif clf_name == "Decision Tree Classifier":
            params["criterion"] = st.sidebar.selectbox('Criterion', ['gini', 'entropy'], index=0)
            params["splitter"] = st.sidebar.selectbox('Splitter', ['best', 'random'], index=0)
            params["max_depth"] = st.sidebar.slider('Max Depth', 1, 100, None, 1)
            params["min_samples_split"] = st.sidebar.slider('Min Samples Split', 2, 20, 2, 1)
            params["min_samples_leaf"] = st.sidebar.slider('Min Samples Leaf', 1, 20, 1, 1)
            params["min_weight_fraction_leaf"] = st.sidebar.slider('Min Weight Fraction Leaf', 0.0, 0.5, 0.0, 0.01)
            params["max_features"] = st.sidebar.selectbox('Max Features', ['auto', 'sqrt', 'log2'], index=0)
            params["max_leaf_nodes"] = st.sidebar.slider('Max Leaf Nodes', 2, 100, None, 1)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["ccp_alpha"] = st.sidebar.slider('CCP Alpha', 0.0, 1.0, 0.0, 0.01)
            class_weight_options = [None, 'balanced']
            params["class_weight"] = st.sidebar.selectbox('Class Weight', class_weight_options, index=0)
        elif clf_name == "Random Forest Classifer":
            params["bootstrap"] = st.sidebar.selectbox('Bootstrap', [True, False], index=0)
            params["ccp_alpha"] = st.sidebar.slider('CCP Alpha', 0.0, 1.0, 0.0, 0.01)
            class_weight_options = [None, 'balanced', 'balanced_subsample']
            params["class_weight"] = st.sidebar.selectbox('Class Weight', class_weight_options, index=0)
            params["criterion"] = st.sidebar.selectbox('Criterion', ['gini', 'entropy'], index=0)
            params["max_depth"] = st.sidebar.slider('Max Depth', 1, 100, None, 1)
            params["max_features"] = st.sidebar.selectbox('Max Features', ['auto', 'sqrt', 'log2', None], index=0)
            params["max_leaf_nodes"] = st.sidebar.slider('Max Leaf Nodes', 2, 100, None, 1)
            params["max_samples"] = st.sidebar.slider('Max Samples', 0.0, 1.0, None, 0.01)
            params["min_impurity_decrease"] = st.sidebar.slider('Min Impurity Decrease', 0.0, 1.0, 0.0, 0.01)
            params["min_samples_leaf"] = st.sidebar.slider('Min Samples Leaf', 1, 20, 1, 1)
            params["min_samples_split"] = st.sidebar.slider('Min Samples Split', 2, 20, 2, 1)
            params["min_weight_fraction_leaf"] = st.sidebar.slider('Min Weight Fraction Leaf', 0.0, 0.5, 0.0, 0.01)
            params["n_estimators"] = st.sidebar.slider('Number of Estimators', 1, 1000, 100, 1)
            params["n_jobs"] = st.sidebar.slider('Number of Jobs', -1, 10, None, 1)
            params["oob_score"] = st.sidebar.checkbox('OOB Score', False)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["verbose"] = st.sidebar.slider('Verbose', 0, 10, 0, 1)
            params["warm_start"] = st.sidebar.checkbox('Warm Start', False)
        elif clf_name == "MultinominalNB":
            params["alpha"] = st.sidebar.slider('Alpha', 0.0, 1.0, 1.0, 0.01)
            params["fit_prior"] = st.sidebar.selectbox('Fit Prior', [True, False], index=0)
            params["class_prior"] = st.sidebar.selectbox('Class Prior', [None, [0.5, 0.5]], index=0)
        elif clf_name == "Bernoulli NB":
            params["alpha"] = st.sidebar.slider("Smoothing Parameter (alpha)", 0.0, 10.0, 1.0, 0.1)
            params["binarize"] = st.sidebar.slider("Binarize Threshold", 0.0, 1.0, 0.0, 0.1)
            params["fit_prior"] = st.sidebar.selectbox("Fit Prior?", [True, False], index=0)
            class_prior_options = [None, [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
            params["class_prior"] = st.sidebar.selectbox("Class Prior", class_prior_options, index=0)
        elif clf_name == "Ridge Classifier":
            params["alpha"] = st.sidebar.slider("Smoothing Parameter (alpha)", 0.0, 10.0, 1.0, 0.1)
            params["class_weight"] = st.sidebar.selectbox("Class Weight", [None, "balanced"], index=0)
            params["copy_X"] = st.sidebar.checkbox("Copy X?", True)
            params["fit_intercept"] = st.sidebar.checkbox("Fit Intercept?", True)
            params["max_iter"] = st.sidebar.slider("Max Number of Iterations", 100, 10000, None, 100)
            params["normalize"] = st.sidebar.selectbox("Normalize?", [True, False], index=0)
            params["positive"] = st.sidebar.checkbox("Positive Only?", False)
            params["random_state"] = st.sidebar.slider("Random State", 0, 100, None, 1)
            params["solver"] = st.sidebar.selectbox("Solver",
                                                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                                                    index=0)
            params["tol"] = st.sidebar.slider("Tolerance", 0.0001, 0.1, 0.001, 0.0001)
        elif clf_name == "Passive Agressive Classifier":
            params["C"] = st.sidebar.slider("C", 0.0, 10.0, 1.0, 0.1)
            params["average"] = st.sidebar.checkbox("Average?", False)
            params["class_weight"] = st.sidebar.selectbox("Class Weight", [None, "balanced"], index=0)
            params["early_stopping"] = st.sidebar.checkbox("Early Stopping?", False)
            params["fit_intercept"] = st.sidebar.checkbox("Fit Intercept?", True)
            params["loss"] = st.sidebar.selectbox("Loss Function", ["hinge", "squared_hinge"], index=0)
            params["max_iter"] = st.sidebar.slider("Max Number of Iterations", 100, 10000, None, 100)
            params["n_iter_no_change"] = st.sidebar.slider("Number of iterations with no improvement to wait", 1, 20,
                                                           None,
                                                           1)
            params["n_jobs"] = st.sidebar.slider("Number of CPU cores to use (-1 for all)", -1, 16, -1, 1)
            params["random_state"] = st.sidebar.slider("Random State", 0, 100, None, 1)
            params["shuffle"] = st.sidebar.checkbox("Shuffle?", True)
            params["tol"] = st.sidebar.slider("Tolerance", 0.0001, 0.1, 0.001, 0.0001)
            params["validation_fraction"] = st.sidebar.slider("Validation Fraction", 0.0, 1.0, 0.1, 0.1)
            params["verbose"] = st.sidebar.slider("Verbose", 0, 2, 0, 1)
            params["warm_start"] = st.sidebar.checkbox("Warm Start?", False)
        elif clf_name == "Linear Regression":
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            params["normalize"] = st.sidebar.checkbox('Normalize', False)
            params["copy_X"] = st.sidebar.checkbox('Copy X', True)
            params["n_jobs"] = st.sidebar.slider('Number of Jobs', -1, 16, None, 1)
        elif clf_name == "Ridge Regression":
            params["alpha"] = st.sidebar.slider('Alpha', 0.01, 100.0, 1.0, 0.01)
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            params["normalize"] = st.sidebar.checkbox('Normalize', False)
            params["copy_X"] = st.sidebar.checkbox('Copy X', True)
            params["max_iter"] = st.sidebar.slider('Max Iterations', 10, 1000, 100, 10)
            params["tol"] = st.sidebar.slider('Tolerance', 0.0001, 0.1, 0.0001, 0.0001)
            params["solver"] = st.sidebar.selectbox('Solver',
                                                    ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                                                    index=0)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
        elif clf_name == "Lasso Regression":
            params["alpha"] = st.sidebar.slider('Alpha', 0.01, 100.0, 1.0, 0.01)
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            params["normalize"] = st.sidebar.checkbox('Normalize', False)
            params["copy_X"] = st.sidebar.checkbox('Copy X', True)
            params["max_iter"] = st.sidebar.slider('Max Iterations', 10, 1000, 100, 10)
            params["tol"] = st.sidebar.slider('Tolerance', 0.0001, 0.1, 0.0001, 0.0001)
            params["selection"] = st.sidebar.selectbox('Selection', ['cyclic', 'random'], index=0)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
        elif clf_name == "ElasticNet Regression":
            params["alpha"] = st.sidebar.slider('Alpha', 0.01, 100.0, 1.0, 0.01)
            params["l1_ratio"] = st.sidebar.slider('L1 Ratio', 0.0, 1.0, 0.5, 0.01)
            params["fit_intercept"] = st.sidebar.checkbox('Fit Intercept', True)
            params["normalize"] = st.sidebar.checkbox('Normalize', False)
            params["copy_X"] = st.sidebar.checkbox('Copy X', True)
            params["max_iter"] = st.sidebar.slider('Max Iterations', 10, 1000, 100, 10)
            params["tol"] = st.sidebar.slider('Tolerance', 0.0001, 0.1, 0.0001, 0.0001)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
        elif clf_name == "Random Forest Regression":
            params["n_estimators"] = st.sidebar.slider('Number of Estimators', 10, 500, 100, 10)
            params["criterion"] = st.sidebar.selectbox('Criterion', ['mse', 'mae'], index=0)
            params["max_depth"] = st.sidebar.slider('Max Depth', 1, 100, None, 1)
            params["min_samples_split"] = st.sidebar.slider('Min Samples Split', 2, 20, 2, 1)
            params["min_samples_leaf"] = st.sidebar.slider('Min Samples Leaf', 1, 20, 1, 1)
            params["max_features"] = st.sidebar.selectbox('Max Features', ['auto', 'sqrt', 'log2'], index=0)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["n_jobs"] = st.sidebar.slider('Number of Jobs', -1, 16, None, 1)
        elif clf_name == "Gradient Boosting Regression":
            params["loss"] = st.sidebar.selectbox('Loss', ['ls', 'lad', 'huber', 'quantile'], index=0)
            params["learning_rate"] = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1, 0.01)
            params["n_estimators"] = st.sidebar.slider('Number of Estimators', 10, 500, 100, 10)
            params["subsample"] = st.sidebar.slider('Subsample', 0.1, 1.0, 1.0, 0.1)
            params["criterion"] = st.sidebar.selectbox('Criterion', ['friedman_mse', 'mse', 'mae'], index=0)
            params["max_depth"] = st.sidebar.slider('Max Depth', 1, 100, None, 1)
            params["min_samples_split"] = st.sidebar.slider('Min Samples Split', 2, 20, 2, 1)
            params["min_samples_leaf"] = st.sidebar.slider('Min Samples Leaf', 1, 20, 1, 1)
            params["max_features"] = st.sidebar.selectbox('Max Features', ['auto', 'sqrt', 'log2'], index=0)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
            params["n_iter_no_change"] = st.sidebar.slider('Number of Iterations with No Change', 1, 20, 10, 1)
        elif clf_name == "XGBoost Regression":
            params["learning_rate"] = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1, 0.01)
            params["n_estimators"] = st.sidebar.slider('Number of Estimators', 10, 500, 100, 10)
            params["max_depth"] = st.sidebar.slider('Max Depth', 1, 100, None, 1)
            params["booster"] = st.sidebar.selectbox('Booster', ['gbtree', 'gblinear', 'dart'], index=0)
            params["gamma"] = st.sidebar.slider('Gamma', 0.0, 10.0, 0.0, 0.1)
            params["subsample"] = st.sidebar.slider('Subsample', 0.1, 1.0, 1.0, 0.1)
            params["colsample_bytree"] = st.sidebar.slider('Column Subsample by Tree', 0.1, 1.0, 1.0, 0.1)
            params["reg_alpha"] = st.sidebar.slider('L1 Regularization (Alpha)', 0.0, 10.0, 0.0, 0.1)
            params["reg_lambda"] = st.sidebar.slider('L2 Regularization (Lambda)', 0.0, 10.0, 1.0, 0.1)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
        elif clf_name == "LightGBM Regression":
            params["learning_rate"] = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1, 0.01)
            params["n_estimators"] = st.sidebar.slider('Number of Estimators', 10, 500, 100, 10)
            params["max_depth"] = st.sidebar.slider('Max Depth', 1, 100, None, 1)
            params["num_leaves"] = st.sidebar.slider('Number of Leaves', 2, 100, 31, 1)
            params["boosting_type"] = st.sidebar.selectbox('Boosting Type', ['gbdt', 'dart', 'goss'], index=0)
            params["subsample"] = st.sidebar.slider('Subsample', 0.1, 1.0, 1.0, 0.1)
            params["colsample_bytree"] = st.sidebar.slider('Column Subsample by Tree', 0.1, 1.0, 1.0, 0.1)
            params["reg_alpha"] = st.sidebar.slider('L1 Regularization (Alpha)', 0.0, 10.0, 0.0, 0.1)
            params["reg_lambda"] = st.sidebar.slider('L2 Regularization (Lambda)', 0.0, 10.0, 1.0, 0.1)
            params["random_state"] = st.sidebar.slider('Random State', 0, 100, None, 1)
        elif clf_name == "ARIMA":
            params["order"] = st.sidebar.text_input('Order (p, d, q)', value="1, 0, 1")
        elif clf_name == "SARIMA":
            params["order"] = st.sidebar.text_input('Order (p, d, q)', value="1, 0, 1")
            params["seasonal_order"] = st.sidebar.text_input('Seasonal Order (P, D, Q, s)', value="1, 0, 1, 12")
        elif clf_name == "Exponential Smoothing":
            params["trend"] = st.sidebar.selectbox('Trend', ['add', 'mul', None], index=0)
            params["seasonal"] = st.sidebar.selectbox('Seasonal', ['add', 'mul', None], index=0)
            params["seasonal_periods"] = st.sidebar.number_input('Seasonal Periods', min_value=1, max_value=100,
                                                                 value=12)
            params["damping_trend"] = st.sidebar.number_input('Damping for Trend', min_value=0.0, max_value=1.0,
                                                              value=1.0)
            params["damping_seasonal"] = st.sidebar.number_input('Damping for Seasonal', min_value=0.0, max_value=1.0,
                                                                 value=1.0)
            params["use_boxcox"] = st.sidebar.checkbox('Use Box-Cox Transformation', value=False)

        return params

    def get_algorithm(self, clf_name, params):
        clf = None
        if clf_name == "Logistic Regression":
            clf = LogisticRegression(
                C=params["C"],
                class_weight=params["class_weight"],
                dual=params["dual"],
                fit_intercept=params["fit_intercept"],
                l1_ratio=params["l1_ratio"],
                max_iter=params["max_iter"],
                multi_class=params["multi_class"],
                n_jobs=params["n_jobs"],
                penalty=params["penalty"],
                random_state=params["random_state"],
                solver=params["solver"],
                tol=params["tol"],
                verbose=params["verbose"],
                warm_start=params["warm_start"]
            )
        elif clf_name == "Perceptron":
            clf = Perceptron(
                alpha=params["alpha"],
                class_weight=params["class_weight"],
                early_stopping=params["early_stopping"],
                eta0=params["eta0"],
                fit_intercept=params["fit_intercept"],
                l1_ratio=params["l1_ratio"],
                max_iter=params["max_iter"],
                n_iter_no_change=params["n_iter_no_change"],
                n_jobs=params["n_jobs"],
                penalty=params["penalty"],
                random_state=params["random_state"],
                shuffle=params["shuffle"],
                tol=params["tol"],
                validation_fraction=params["validation_fraction"],
                verbose=params["verbose"],
                warm_start=params["warm_start"]
            )
        elif clf_name == "Linear SVC":
            clf = LinearSVC(
                C=params["C"],
                class_weight=params["class_weight"],
                dual=params["dual"],
                fit_intercept=params["fit_intercept"],
                intercept_scaling=params["intercept_scaling"],
                loss=params["loss"],
                max_iter=params["max_iter"],
                multi_class=params["multi_class"],
                penalty=params["penalty"],
                random_state=params["random_state"],
                tol=params["tol"],
                verbose=params["verbose"]
            )

        elif clf_name == "Decision Tree Classifier":
            clf = DecisionTreeClassifier(
                criterion=params["criterion"],
                splitter=params["splitter"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                max_features=params["max_features"],
                max_leaf_nodes=params["max_leaf_nodes"],
                random_state=params["random_state"],
                ccp_alpha=params["ccp_alpha"],
                class_weight=params["class_weight"]
            )
        elif clf_name == "Random Forest Classifer":
            clf = RandomForestClassifier(
                bootstrap=params["bootstrap"],
                ccp_alpha=params["ccp_alpha"],
                class_weight=params["class_weight"],
                criterion=params["criterion"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                max_leaf_nodes=params["max_leaf_nodes"],
                max_samples=params["max_samples"],
                min_impurity_decrease=params["min_impurity_decrease"],
                min_samples_leaf=params["min_samples_leaf"],
                min_samples_split=params["min_samples_split"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                n_estimators=params["n_estimators"],
                n_jobs=params["n_jobs"],
                oob_score=params["oob_score"],
                random_state=params["random_state"],
                verbose=params["verbose"],
                warm_start=params["warm_start"]
            )

        elif clf_name == "MultinominalNB":
            clf = MultinomialNB(
                alpha=params["alpha"],
                fit_prior=params["fit_prior"],
                class_prior=params["class_prior"]
            )
        elif clf_name == "Bernoulli NB":
            clf = BernoulliNB(
                alpha=params["alpha"],
                binarize=params["binarize"],
                fit_prior=params["fit_prior"],
                class_prior=params["class_prior"]
            )
        elif clf_name == "Ridge Classifier":
            clf = RidgeClassifier(
                alpha=params["alpha"],
                class_weight=params["class_weight"],
                copy_X=params["copy_X"],
                fit_intercept=params["fit_intercept"],
                max_iter=params["max_iter"],
                normalize=params["normalize"],
                positive=params["positive"],
                random_state=params["random_state"],
                solver=params["solver"],
                tol=params["tol"]
            )
        elif clf_name == "Passive Agressive Classifier":
            clf = PassiveAggressiveClassifier(
                C=params["C"],
                average=params["average"],
                class_weight=params["class_weight"],
                early_stopping=params["early_stopping"],
                fit_intercept=params["fit_intercept"],
                loss=params["loss"],
                max_iter=params["max_iter"],
                n_iter_no_change=params["n_iter_no_change"],
                n_jobs=params["n_jobs"],
                random_state=params["random_state"],
                shuffle=params["shuffle"],
                tol=params["tol"],
                validation_fraction=params["validation_fraction"],
                verbose=params["verbose"],
                warm_start=params["warm_start"]
            )

        elif clf_name == "Linear Regression":
            clf = LinearRegression(
                fit_intercept=params["fit_intercept"],
                normalize=params["normalize"],
                copy_X=params["copy_X"],
                n_jobs=params["n_jobs"]
            )
        elif clf_name == "Ridge Regression":
            clf = Ridge(
                alpha=params["alpha"],
                fit_intercept=params["fit_intercept"],
                normalize=params["normalize"],
                copy_X=params["copy_X"],
                max_iter=params["max_iter"],
                tol=params["tol"],
                solver=params["solver"],
                random_state=params["random_state"]
            )
        elif clf_name == "Lasso Regression":
            clf = Lasso(
                alpha=params["alpha"],
                fit_intercept=params["fit_intercept"],
                normalize=params["normalize"],
                copy_X=params["copy_X"],
                max_iter=params["max_iter"],
                tol=params["tol"],
                selection=params["selection"]
            )
        elif clf_name == "ElasticNet Regression":
            clf = ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                fit_intercept=params["fit_intercept"],
                normalize=params["normalize"],
                copy_X=params["copy_X"],
                max_iter=params["max_iter"],
                tol=params["tol"],
                random_state=params["random_state"]
            )
        elif clf_name == "Support Vector Regression (SVR)":
            clf = SVR(
                kernel=params["kernel"],
                degree=params["degree"],
                gamma=params["gamma"],
                C=params["C"],
                epsilon=params["epsilon"],
                shrinking=params["shrinking"],
                tol=params["tol"],
                max_iter=params["max_iter"]
            )
        elif clf_name == "Decision Tree Regression":
            clf = DecisionTreeRegressor(
                criterion=params["criterion"],
                splitter=params["splitter"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                max_features=params["max_features"],
                max_leaf_nodes=params["max_leaf_nodes"],
                random_state=params["random_state"],
                ccp_alpha=params["ccp_alpha"],
                class_weight=params["class_weight"]
            )
        elif clf_name == "Random Forest Regression":
            clf = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                criterion=params["criterion"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                max_features=params["max_features"],
                max_leaf_nodes=params["max_leaf_nodes"],
                min_impurity_decrease=params["min_impurity_decrease"],
                bootstrap=params["bootstrap"],
                oob_score=params["oob_score"],
                random_state=params["random_state"]
            )
        elif clf_name == "Gradient Boosting Regression":
            clf = GradientBoostingRegressor(
                loss=params["loss"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                subsample=params["subsample"],
                criterion=params["criterion"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                random_state=params["random_state"]
            )
        elif clf_name == "AdaBoost Regression":
            clf = AdaBoostRegressor(
                base_estimator=params["base_estimator"],
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                loss=params["loss"]
            )
        elif clf_name == "XGBoost Regression":
            clf = XGBRegressor(
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                objective=params["objective"],
                booster=params["booster"],
                gamma=params["gamma"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                random_state=params["random_state"],
                verbosity=params["verbosity"]
            )
        elif clf_name == "ARIMA":
            clf = ARIMA(
                order=params["order"]
            )
        elif clf_name == "SARIMA":
            clf = SARIMAX(
                order=params["order"],
                seasonal_order=params["seasonal_order"]
            )
        elif clf_name == "Exponential Smoothing":
            clf = ExponentialSmoothing(
                trend=params["trend"],
                seasonal=params["seasonal"],
                seasonal_periods=params["seasonal_periods"],
                damping_trend=params["damping_trend"],
                damping_seasonal=params["damping_seasonal"],
                use_boxcox=params["use_boxcox"]
            )

        return clf

    def train_model(self, clf, X_train, y_train):
        pass

    def evaluate_model(self, clf, X_test, y_test):
        pass

    def export_model_and_metadata(self):
        pass

class Dashboard:
    def __init__(self, model_file):
        pass

    def load_model(self, file):
        model = pickle.load(file)
        return model

    @staticmethod
    def show_metrics(acc, y_test, y_pred, classifier):
        st.write(f"Accuracy = {acc}")
        st.progress(acc)
        st.write('confusion matrix : ')
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"))
        st.plotly_chart(fig)

        st.write("Download the trained model")

        model_download = st.button("Download")
        if model_download:
            filename = f"{classifier}.pkl"
            with open(filename, 'wb') as file:
                pickle.dump(classifier, file)


def app():
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        with st.sidebar.expander('preprdocessing'):
            st.sidebar.title('Data Preprocessing')
            missing_values_threshold = st.sidebar.slider("Missing values threshold", min_value=0.0, max_value=1.0,
                                                         step=0.05, value=0.25)
            target_variable = st.sidebar.text_input("Target variable")
            test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
            random_state = st.sidebar.number_input("Random state", value=25)

        Cleaning_instance = DataPreprocessing(file)
        data = Cleaning_instance.data

        cleaning_options = ['handle missing values', 'encode categorical variables', 'scale numerical variables',
                            'split data']

        with st.expander("Clean"):
            selected_option = st.multiselect('Select option:', cleaning_options)

            if 'handle missing values' in selected_option:
                data = Cleaning_instance.handle_missing_values(missing_values_threshold)

            if 'encode categorical variables' in selected_option:
                data = Cleaning_instance.encode_categorical_variables(data)
            if 'scale numerical variables' in selected_option:
                data = Cleaning_instance.scale_numeric_variables(data, target_variable)

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
                    X_train, X_test, y_train, y_test = Cleaning_instance.split_data(data, target_variable, test_size,
                                                                                    random_state)
                    train_dataset = X_train.to_csv(index=True).encode('utf-8')
                    test_dataset = X_test.to_csv(index=True).encode('utf-8')
                    train_variable = y_train.to_csv(index=True).encode('utf-8')
                    test_variable = y_test.to_csv(index=True).encode('utf-8')

                    st.write("Training Data")
                    st.table(X_train.head())
                    st.write("Testing Data")
                    st.table(X_test.head())

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.write("X_train : ")
                        st.download_button(
                            "Download",
                            train_dataset,
                            "X_train.csv",
                            "text/csv",
                            key='download1-csv'
                        )
                    with c2:
                        st.write("X_test : ")
                        st.download_button(
                            "Download",
                            test_dataset,
                            "X_test.csv",
                            "text/csv",
                            key='download2-csv'
                        )
                    with c3:
                        st.write("y_train : ")
                        st.download_button(
                            "Download",
                            train_variable,
                            "Y_train.csv",
                            "text/csv",
                            key='download3-csv'
                        )
                    with c4:
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
            ('None', 'Classification', 'Regression', 'Forecasting', 'Sentiment Analysis'))

        Model_instance = ModelBuilding(data, target_variable)

        if ml_problem == 'None':
            pass
        elif ml_problem == 'Classification':

            st.sidebar.subheader("Select Classifier")
            algorithms = (
                "Logistic Regression", "Perceptron", "Linear SVC", "Decision Tree Classifier",
                "MultinominalNB", "Bernoulli NB", "Ridge Classifier", "Passive Agressive Classifier")
            problem_type = st.sidebar.selectbox("Classifier", algorithms)

            params = Model_instance.add_parameter_ui(problem_type)
            classifier = Model_instance.get_algorithm(problem_type, params)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            training_expander = st.expander("Train")
            with training_expander:
                training_expander.title('TRAINING')

                Dashboard.show_metrics(acc, y_test, y_pred, classifier)


        elif ml_problem == 'Regression':

            st.sidebar.subheader("Select Regression")
            algorithms = (
                "Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                "Random Forest Regression", "Gradient Boosting Regression", "XGBoost Regression", "LightGBM Regression")
            problem_type = st.sidebar.selectbox("Regression", algorithms)

            params = Model_instance.add_parameter_ui(problem_type)
            classifier = Model_instance.get_algorithm(problem_type, params)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            training_expander = st.expander("Train")
            with training_expander:
                training_expander.title('TRAINING')

                Dashboard.show_metrics(acc, y_test, y_pred, classifier)

        elif ml_problem == 'Forecasting':
            st.sidebar.subheader("Select Forecaster")
            algorithms = (
                "ARIMA", "SARIMA", "Exponential Smoothing")
            problem_type = st.sidebar.selectbox("Forecasting", algorithms)

            params = Model_instance.add_parameter_ui(problem_type)
            classifier = Model_instance.get_algorithm(problem_type, params)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            training_expander = st.expander("Train")
            with training_expander:
                training_expander.title('TRAINING')

                Dashboard.show_metrics(acc, y_test, y_pred, classifier)

        elif ml_problem == 'Sentiment Analysis':
            pass

        st.sidebar.title('PREDICTION')

        model_file = st.sidebar.file_uploader('Upload a trained model', type="pkl")
        prediction_expander = st.expander("Predict")

        Dashboard_instance = Dashboard(model_file)

        with prediction_expander:
            if model_file is not None:

                st.title('PREDICTION')

                col1, col2, col3 = st.columns(3)
                model = Dashboard_instance.load_model(model_file)
                model_features = model.feature_names_in_
                new_data_record = []
                i = 0
                for feature in model_features:
                    if i == 0:
                        n = col1.number_input(feature)
                        i = 1
                    elif i == 1:
                        n = col2.number_input(feature)
                        i = 2
                    else:
                        n = col3.number_input(feature)
                        i = 0
                    # n = st.text_input(feature)
                    new_data_record.append(n)
                predict_button = st.button('predict')
                if predict_button:
                    prediction_result = model.predict([new_data_record])
                    if prediction_result == 1:
                        st.header('Positive result')
                    elif prediction_result == 0:
                        st.header('Negative result')
                    else:
                        st.write('Neutral / Other')
    else:
        with st.expander("Usage guide"):
            st.title("Usage guide")
            st.write("step 1 : upload a dataset in csv format")
            st.write("step 2 : try the diffrenet options of data preprocessing")
            st.write("step 3 : select the target variable")
            st.write("step 4 : download the test and training data")
            st.write("step 5 : select the right problem classification , ...")
            st.write("step 6 : select the right algorithm")
            st.write("step 7 : change the hyperparameters and tune them for better results")
            st.write("step 8 : download the trained model")
            st.write("step 9 : upload a model in a pickle format")
            st.write("step 10 : fill the input fields and click predict")
        ut.centered_title("Machine Learning Dashboard")



if __name__ == "__main__":
    app()
