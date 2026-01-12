from zenml.steps import step
import pandas as pd

@step
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/food.csv")
    return df