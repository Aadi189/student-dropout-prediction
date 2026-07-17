# src/models.py
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_model_suite():
    """
    Returns the collection of core evaluation models.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

def save_model(model, model_name, save_dir="/content/drive/MyDrive/model_results"):
    """
    Saves the trained model object using pickle.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"best_{model_name.replace(' ', '_').lower()}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved {model_name} model to: {filename}")
