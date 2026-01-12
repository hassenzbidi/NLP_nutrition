from zenml.steps import step
from scripts.recommender import NutritionRecommender

@step
def train_model(df):
    model = NutritionRecommender()
    return model

import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight = trial.suggest_float("weight", 0.1, 1.0)
    score = 1 / (abs(lr - 0.01) + weight)  # exemple
    return score

@step
def train_model(df):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(study.best_params)
    return study.best_params

import mlflow
mlflow.log_params(study.best_params)