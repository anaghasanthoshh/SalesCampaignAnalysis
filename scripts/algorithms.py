import pandas as pd
import mlflow
import os
from config.config import processed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import pandas as pd
from imblearn.over_sampling import SMOTE



def loading_data(processed_data,train_file_='processed_train.json',test_file='processed_test.json'):
    df_train=pd.read_json(os.path.join(processed_data,train_file_))
    df_test=pd.read_json(os.path.join(processed_data,test_file))
    X_test=df_test.drop('Conversion',axis=1)
    y_test=df_test['Conversion']
    X_train=df_train.drop('Conversion',axis=1)
    y_train=df_train['Conversion']
    print(f"X_train shape:{X_train.shape}\n y_train shape :{y_train.shape}")
    return X_test,y_test,X_train,y_train

#SMOTE to resolve imbalanced data
def smote(X_train,y_train):
    #SMOTE to balance the data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled,y_train_resampled

#Random Forest
def random_forest(X_train,y_train):
    rf_classifier=RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    rf_classifier.fit(X_train,y_train)
    return rf_classifier

def evaluate_model(y_test, y_predict):
    report = classification_report(y_test, y_predict,output_dict=True)
    accuracy = accuracy_score(y_test, y_predict)
    cm=confusion_matrix(y_test, y_predict)
    return accuracy, report, cm

def log_results(experiment_name, run_name, X_train, y_train, X_test, y_test, use_smote=False):
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log whether SMOTE was used
        mlflow.log_param("SMOTE Applied", use_smote)

        # Train the model
        rf_classifier = random_forest(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        # Evaluate the model
        accuracy, report, cm = evaluate_model(y_test, y_pred)

        # Log model parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_high", report['High']['precision'])
        mlflow.log_metric("recall_high", report['High']['recall'])
        mlflow.log_metric("f1_high", report['High']['f1-score'])
        mlflow.log_metric("precision_low", report['Low']['precision'])
        mlflow.log_metric("recall_low", report['Low']['recall'])
        mlflow.log_metric("f1_low", report['Low']['f1-score'])
        mlflow.log_metric("precision_medium", report['Medium']['precision'])
        mlflow.log_metric("recall_medium", report['Medium']['recall'])
        mlflow.log_metric("f1_medium", report['Medium']['f1-score'])

        # Log confusion matrix as an artifact
        cm_path = "confusion_matrix.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        mlflow.log_artifact(cm_path)

        # Print the evaluation metrics
        print(f"Run: {run_name} - Accuracy: {accuracy:.2f}")


X_test,y_test,X_train,y_train=loading_data(processed_data)
#X_train_resampled,y_train_resampled=smote(X_train,y_train)
#accuracy, report, cm= random_forest(X_train_resampled,y_train_resampled,X_test, y_test)
log_results('Sales Campaign Analysis','RFC_No_SMOTE',X_train, y_train, X_test, y_test)
