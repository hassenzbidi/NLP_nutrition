from zenml import pipeline
from steps.load_data import load_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def nutrition_pipeline():
    data = load_data()
    model = train_model(data)
    evaluate_model(model, data)