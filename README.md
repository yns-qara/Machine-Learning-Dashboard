# Machine-Learning-Dashboard

This dashboard is a machine learning model training and evaluation tool that allows users to upload a CSV dataset, preprocess it with options for handling missing values, encoding categorical variables, and scaling numeric variables. It also allows users to split the data into train and test sets and select a target variable. Users can choose from three classifiers, Random Forest, SVM, or Logistic Regression, and set their hyperparameters through a sidebar UI. The dashboard then trains the selected model and displays its accuracy score and confusion matrix. Finally, the user can download the preprocessed data, the model, and its predictions on the test set.

to get started :

1. clone this reposotory

  ```shell
  git clone https://github.com/yns-qara/Machine-Learning-Dashboard.git
  ```
2. open the directory where the source code exists

  ```shell
  cd Machine-Learning-Dashboard
  ```
  
3. install the requirements

  ```shell
  pip install -r requirements.txt
  ```
4. run the app

  ```shell
  streamlit run CombinedDashboard.py
  ```
