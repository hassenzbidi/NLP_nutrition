import streamlit as st
import os
import time
import mlflow
import pandas as pd
from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM
from scripts.recommender import NutritionRecommender
from contextlib import nullcontext
import requests

# =========================================================
# CONFIGURATION MLFLOW (ROBUSTE AVEC RETRY)
# =========================================================

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

def wait_for_mlflow(uri, retries=12, delay=2):
    for i in range(retries):
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.search_experiments()
            print("✅ MLflow connecté")
            return True
        except Exception as e:
            print(f"[MLFLOW] Tentative {i+1}/{retries} échouée : {e}")
            time.sleep(delay)
    return False

mlflow_enabled = wait_for_mlflow(MLFLOW_URI)

if mlflow_enabled:
    mlflow.set_experiment("Nutrition_Assistant_Project")

# =========================================================
# CONFIGURATION STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Expert Nutrition IA - v2",
    layout="wide"
)

# =========================================================
# CHARGEMENT DES RESSOURCES (CACHE)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_resources():
    recommender = NutritionRecommender()

    usda_data = None
    if os.path.exists("data/food.csv"):
        usda_data = pd.read_csv("data/food.csv").head(10)

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_type="mistral",
        context_length=2048
    )

    return recommender, tokenizer, model, usda_data

# =========================================================
# INITIALISATION
# =========================================================

if "ready" not in st.session_state:
    with st.status("🚀 Initialisation du système...", expanded=True) as status:
        st.session_state.data = load_resources()
        st.session_state.ready = True
        status.update(label="✅ Système prêt !", state="complete")

recommender, tokenizer, model, usda_data = st.session_state.data

# =========================================================
# INTERFACE UTILISATEUR
# =========================================================

with st.sidebar:
    st.title("Assistant Nutrition")

    if mlflow_enabled:
        st.success("🟢 Tracking MLflow actif")
    else:
        st.warning("⚠️ Tracking désactivé (MLflow indisponible)")

    st.info(f"Version : {'v2 (USDA)' if usda_data is not None else 'v1 (Baseline)'}")

    condition = st.text_input(
        "Condition médicale",
        placeholder="ex: Anémie"
    )

    analyze_btn = st.button("Lancer l'analyse", type="primary")

# =========================================================
# LOGIQUE D'ANALYSE
# =========================================================

if analyze_btn:
    if not condition:
        st.warning("Veuillez entrer une condition médicale.")
    else:
        run_ctx = mlflow.start_run() if mlflow_enabled else nullcontext()

        with run_ctx:
            with st.spinner("Analyse en cours..."):
                start_t = time.time()

                res = recommender.get_recommendations(
                    {"condition": condition}
                )

                prompt = (
                    f"<s>[INST] Expert Nutrition. "
                    f"Fait: {res['summary']}. "
                    f"Conseil pour: {condition} [/INST]</s>"
                )

                answer = model(prompt, max_new_tokens=250)

                duration = time.time() - start_t

                if mlflow_enabled:
                    mlflow.log_param("condition", condition)
                    mlflow.log_metric("latency", duration)

                st.subheader("💡 Conseil de l'IA")
                st.write(answer)
                st.success(f"Analyse terminée en {duration:.2f}s")

        if usda_data is not None:
            with st.expander("📊 Données USDA v2 utilisées"):
                st.dataframe(usda_data)

# =========================================================
# DEBUG OPTIONNEL
# =========================================================

with st.expander("🔧 Diagnostic MLflow"):
    try:
        r = requests.get("http://mlflow-server:5000")
        st.success("MLflow accessible depuis Streamlit")
    except Exception as e:
        st.error(f"Erreur MLflow : {e}")
