import argparse
import csv
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the heart disease model and export Vercel artifacts."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the training CSV. It must contain the 13 feature columns and target.",
    )
    parser.add_argument(
        "--target",
        default="target",
        help="Target column name. Defaults to target.",
    )
    parser.add_argument(
        "--out",
        default="model",
        help="Directory where model.pkl and scaler.pkl will be written.",
    )
    return parser.parse_args()


def load_dataset(csv_path, target_column):
    rows = []
    labels = []

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        required = set(FEATURE_ORDER + [target_column])
        missing = sorted(required - columns)

        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(reader.fieldnames or [])}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[name]) for name in FEATURE_ORDER])
                labels.append(int(float(row[target_column])))
            except ValueError as error:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from error

    if not rows:
        raise ValueError("CSV contains no training rows.")

    return rows, labels


def main():
    args = parse_args()
    rows, labels = load_dataset(args.csv, args.target)

    x_train, x_test, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=5,
        min_samples_split=10,
    )
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    probabilities = model.predict_proba(x_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)

    with (output_dir / "scaler.pkl").open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"Saved {output_dir / 'model.pkl'}")
    print(f"Saved {output_dir / 'scaler.pkl'}")
    print(f"Holdout accuracy: {accuracy:.3f}")
    print(f"Holdout ROC AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
