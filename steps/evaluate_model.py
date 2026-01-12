from zenml.steps import step

@step
def evaluate_model(model, df):
    print("Évaluation simple : modèle chargé avec succès ✔️")