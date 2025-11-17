import os
import sys
import mlflow
import mlflow.sklearn

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

#os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/niranjancharan75/networksecurity.mlflow"
#os.environ["MLFLOW_TRACKING_USERNAME"]="niranjancharan75"
#os.environ["MLFLOW_TRACKING_PASSWORD"]="3df3db75c09d7a42b7836d9a53204ab069269021"


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            # Set MLflow tracking URI here once
            mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")
            logging.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

        except Exception as e:
            raise CustomException(e, sys)

    def track_mlflow(self, best_model, train_metrics: dict, test_metrics: dict):
        """
        Logs parameters, metrics and model to MLflow in a single run.
        """
        with mlflow.start_run():
            # Log training metrics with prefix 'train_'
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)

            # Log testing metrics with prefix 'test_'
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            # Log the model itself
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            logging.info("MLflow run logged successfully.")

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
                "Random Forest": {'n_estimators': [8, 16, 32, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models with hyperparameter tuning
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                models=models, param=params
            )

            # Select best model based on highest score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Train the best model
            best_model.fit(X_train, y_train)

            # Get predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(x_test)

            # Get classification metrics as dictionaries for easy logging
            train_metrics_obj = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            train_metrics = {
                "f1_score": train_metrics_obj.f1_score,
                "precision": train_metrics_obj.precision_score,
                "recall": train_metrics_obj.recall_score,
            }

            test_metrics_obj = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            test_metrics = {
                "f1_score": test_metrics_obj.f1_score,
                "precision": test_metrics_obj.precision_score,
                "recall": test_metrics_obj.recall_score,
            }

            # Track all info in one MLflow run
            self.track_mlflow(best_model, train_metrics, test_metrics)

            # Load preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Save the combined model (preprocessor + model)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            # Also save final model separately if needed
            save_object("final_model/model.pkl", best_model)

            # Prepare and return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics_obj,
                test_metric_artifact=test_metrics_obj
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load numpy arrays for train and test
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split features and labels
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train model and return artifact
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
