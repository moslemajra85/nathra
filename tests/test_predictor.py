import os
import tempfile
import unittest

from predictor import ModelNotConfigured, ValidationError, load_artifacts, predict, validate_payload


VALID_PAYLOAD = {
    "age": 50,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 0,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


class PredictorValidationTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("MODEL_DIR", None)
        load_artifacts.cache_clear()

    def test_valid_payload_is_normalized(self):
        payload = dict(VALID_PAYLOAD)
        payload["oldpeak"] = "1.5"

        values = validate_payload(payload)

        self.assertEqual(values["sex"], 1)
        self.assertEqual(values["oldpeak"], 1.5)

    def test_missing_required_field_reports_specific_error(self):
        payload = dict(VALID_PAYLOAD)
        del payload["age"]

        with self.assertRaises(ValidationError) as context:
            validate_payload(payload)

        self.assertIn("age", context.exception.errors)

    def test_out_of_range_category_is_rejected(self):
        payload = dict(VALID_PAYLOAD)
        payload["thal"] = 9

        with self.assertRaises(ValidationError) as context:
            validate_payload(payload)

        self.assertIn("thal", context.exception.errors)

    def test_predict_reports_missing_artifacts_cleanly(self):
        with tempfile.TemporaryDirectory() as model_dir:
            os.environ["MODEL_DIR"] = model_dir
            load_artifacts.cache_clear()

            with self.assertRaises(ModelNotConfigured):
                predict(VALID_PAYLOAD)


if __name__ == "__main__":
    unittest.main()
