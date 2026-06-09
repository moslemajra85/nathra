from functools import lru_cache
import json
import os
from pathlib import Path


FEATURES = [
    {"name": "age", "label": "Age", "type": "number", "min": 1, "max": 120},
    {"name": "sex", "label": "Sex", "type": "category", "allowed": [0, 1]},
    {"name": "cp", "label": "Chest pain type", "type": "category", "allowed": [0, 1, 2, 3]},
    {"name": "trestbps", "label": "Resting blood pressure", "type": "number", "min": 60, "max": 250},
    {"name": "chol", "label": "Cholesterol", "type": "number", "min": 80, "max": 700},
    {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dl", "type": "category", "allowed": [0, 1]},
    {"name": "restecg", "label": "Resting ECG", "type": "category", "allowed": [0, 1, 2]},
    {"name": "thalach", "label": "Maximum heart rate", "type": "number", "min": 40, "max": 230},
    {"name": "exang", "label": "Exercise induced angina", "type": "category", "allowed": [0, 1]},
    {"name": "oldpeak", "label": "ST depression", "type": "number", "min": 0, "max": 10},
    {"name": "slope", "label": "ST slope", "type": "category", "allowed": [0, 1, 2]},
    {"name": "ca", "label": "Major vessels", "type": "category", "allowed": [0, 1, 2, 3, 4]},
    {"name": "thal", "label": "Thalassemia", "type": "category", "allowed": [0, 1, 2, 3]},
]

FEATURE_ORDER = [feature["name"] for feature in FEATURES]


class ModelNotConfigured(RuntimeError):
    pass


class ValidationError(ValueError):
    def __init__(self, errors):
        super().__init__("Invalid prediction input")
        self.errors = errors


def _model_dir():
    return Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))


def model_paths():
    model_dir = _model_dir()
    return {
        "model": Path(os.environ.get("MODEL_PATH", model_dir / "model.pkl")),
        "scaler": Path(os.environ.get("SCALER_PATH", model_dir / "scaler.pkl")),
        "demo_metadata": model_dir / "demo_model.json",
    }


def model_metadata():
    metadata_path = model_paths()["demo_metadata"]
    if not metadata_path.exists():
        return {
            "mode": "trained",
            "warning": None,
        }

    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@lru_cache(maxsize=1)
def load_artifacts():
    paths = model_paths()
    required_paths = {"model": paths["model"], "scaler": paths["scaler"]}
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        expected = {name: str(path) for name, path in required_paths.items()}
        raise ModelNotConfigured(
            f"Missing {', '.join(missing)} artifact(s). Expected files: {expected}"
        )

    return required_paths


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValidationError({"body": "JSON body must be an object."})

    values = {}
    errors = {}

    for feature in FEATURES:
        name = feature["name"]
        raw_value = payload.get(name)

        if raw_value is None or raw_value == "":
            errors[name] = f"{feature['label']} is required."
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors[name] = f"{feature['label']} must be a number."
            continue

        if feature["type"] == "category":
            if not value.is_integer():
                errors[name] = f"{feature['label']} must be one of {feature['allowed']}."
                continue
            value = int(value)
            if value not in feature["allowed"]:
                errors[name] = f"{feature['label']} must be one of {feature['allowed']}."
                continue
        else:
            minimum = feature.get("min")
            maximum = feature.get("max")
            if minimum is not None and value < minimum:
                errors[name] = f"{feature['label']} must be at least {minimum}."
                continue
            if maximum is not None and value > maximum:
                errors[name] = f"{feature['label']} must be at most {maximum}."
                continue

        values[name] = value

    if errors:
        raise ValidationError(errors)

    return values


def risk_message(prediction, probability):
    if prediction == 1:
        if probability >= 0.8:
            return {
                "level": "Very high risk",
                "tone": "critical",
                "recommendation": "Immediate medical consultation is strongly recommended.",
            }
        if probability >= 0.6:
            return {
                "level": "High risk",
                "tone": "warning",
                "recommendation": "Schedule a doctor's appointment soon.",
            }
        return {
            "level": "Moderate-high risk",
            "tone": "warning",
            "recommendation": "Consider lifestyle changes and regular monitoring.",
        }

    if probability <= 0.2:
        return {
            "level": "Very low risk",
            "tone": "positive",
            "recommendation": "Keep up healthy habits and regular check-ups.",
        }

    return {
        "level": "Low risk",
        "tone": "positive",
        "recommendation": "Continue regular preventive check-ups.",
    }


def demo_probability(values):
    score = -4.0
    score += (values["age"] - 50) * 0.035
    score += 0.35 if values["sex"] == 1 else 0
    score += [0.45, 0.2, -0.15, 0.65][values["cp"]]
    score += (values["trestbps"] - 120) * 0.012
    score += (values["chol"] - 200) * 0.006
    score += 0.35 if values["fbs"] == 1 else 0
    score += 0.2 if values["restecg"] > 0 else 0
    score += (145 - values["thalach"]) * 0.018
    score += 0.8 if values["exang"] == 1 else 0
    score += values["oldpeak"] * 0.45
    score += [0, 0.45, 0.75][values["slope"]]
    score += values["ca"] * 0.55
    score += [0, 0.35, 0.5, 0.25][values["thal"]]

    # Logistic transform without importing numpy/scikit-learn.
    probability = 1 / (1 + pow(2.718281828459045, -score))
    return max(0.01, min(0.99, probability))


def predict(payload):
    values = validate_payload(payload)
    artifact_metadata = model_metadata()

    if artifact_metadata.get("mode") != "demo":
        load_artifacts()
        raise ModelNotConfigured(
            "The Vercel demo API only supports demo_model.json. "
            "Deploy real scikit-learn model inference on Hugging Face, Render, or another Python ML backend."
        )

    probability = demo_probability(values)
    prediction = int(probability >= 0.5)
    risk = risk_message(prediction, probability)
    demo_disclaimer = (
        "This is a synthetic demo model for app development only. It is not medically valid."
        if artifact_metadata.get("mode") == "demo"
        else None
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "probabilityPercent": round(probability * 100, 1),
        "risk": risk,
        "model": artifact_metadata,
        "features": values,
        "disclaimer": demo_disclaimer
        or (
            "This educational prediction is not a diagnosis and must not replace "
            "professional medical advice."
        ),
    }


def metadata():
    paths = model_paths()
    demo_configured = paths["demo_metadata"].exists()
    artifact_metadata = model_metadata() if demo_configured else {"mode": "missing", "warning": None}
    return {
        "status": "ready" if demo_configured else "model_not_configured",
        "model": artifact_metadata,
        "featureOrder": FEATURE_ORDER,
        "features": FEATURES,
        "artifacts": {
            "model": str(paths["model"]),
            "scaler": str(paths["scaler"]),
            "demo_metadata": str(paths["demo_metadata"]),
        },
    }
