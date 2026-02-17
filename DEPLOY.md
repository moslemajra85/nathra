# 🚀 Deploy Heart Disease Model to Hugging Face (No Server Needed!)

This guide will help you deploy your ML model to Hugging Face Spaces for **FREE** - no backend server required!

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Step 1: Export Model from Notebook](#-step-1-export-model-from-notebook)
3. [Step 2: Create Hugging Face Account](#-step-2-create-hugging-face-account)
4. [Step 3: Create a New Space](#-step-3-create-a-new-space)
5. [Step 4: Upload Files](#-step-4-upload-files)
6. [Step 5: Wait for Build](#-step-5-wait-for-build)
7. [Step 6: Get Your API URL](#-step-6-get-your-api-url)
8. [Step 7: Use in Lovable App](#-step-7-use-in-lovable-app)
9. [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

### What We're Doing

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                                              │
│   │ 1. NOTEBOOK  │  Export model.pkl + scaler.pkl               │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │ 2. HUGGING   │  Upload 5 files:                             │
│   │    FACE      │  - app.py                                    │
│   │    SPACES    │  - requirements.txt                          │
│   │              │  - README.md                                  │
│   │              │  - model.pkl                                  │
│   │              │  - scaler.pkl                                 │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │ 3. AUTOMATIC │  Hugging Face builds your app                │
│   │    API!      │  You get FREE API endpoint!                  │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │ 4. LOVABLE   │  Call the API from your frontend             │
│   │    APP       │  No backend server needed! 🎉                │
│   └──────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Hugging Face?

| Feature       | Benefit                               |
| ------------- | ------------------------------------- |
| **Free**      | No cost for CPU basic tier            |
| **No Server** | They host everything for you          |
| **Auto API**  | Gradio gives you an API automatically |
| **Easy**      | Just upload files, done!              |

---

## 📓 Step 1: Export Model from Notebook

### Open your `PIM_Finale_(1).ipynb` notebook

Add a **NEW CELL** at the end and paste this code:

```python
# =====================================================
# EXPORT MODEL FOR HUGGING FACE DEPLOYMENT
# Run this AFTER you have trained your model
# =====================================================

import pickle

# ---------------------------------------------------
# Save the model and scaler
# ---------------------------------------------------
print("Saving model and scaler...")

# Save the scaler (VERY IMPORTANT!)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: scaler.pkl")

# Save the Random Forest model
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Saved: model.pkl")

# ---------------------------------------------------
# Test that the files work
# ---------------------------------------------------
print("\n🔍 Testing the saved files...")

# Load them back
test_model = pickle.load(open('model.pkl', 'rb'))
test_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Test prediction
test_features = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
test_scaled = test_scaler.transform(test_features)
test_pred = test_model.predict(test_scaled)
test_prob = test_model.predict_proba(test_scaled)

print(f"Test prediction: {test_pred[0]}")
print(f"Test probability: {test_prob[0][1]:.2%}")
print("\n✅ SUCCESS! Files are working correctly.")

# ---------------------------------------------------
# If using Google Colab, download the files
# ---------------------------------------------------
print("\n📥 Downloading files...")
try:
    from google.colab import files
    files.download('model.pkl')
    files.download('scaler.pkl')
    print("✅ Files downloaded!")
except:
    print("Not in Colab. Find the files in your working directory.")
```

### Run this cell

After running:

- You'll have **2 files**: `model.pkl` and `scaler.pkl`
- If using Google Colab, they will download automatically
- If using local Jupyter, find them in your notebook's folder

---

## 👤 Step 2: Create Hugging Face Account

1. Go to: **https://huggingface.co/**

2. Click **"Sign Up"** (top right corner)

   ![Sign Up Button](https://i.imgur.com/example.png)

3. Fill in your details:
   - Username (this will be in your API URL!)
   - Email
   - Password

4. Click **"Create Account"**

5. **Verify your email** (check inbox)

6. **Done!** You now have a Hugging Face account

---

## 🆕 Step 3: Create a New Space

1. **Log in** to Hugging Face

2. Click your **profile picture** (top right)

3. Click **"New Space"**

4. Fill in the form:

   | Field          | What to Enter             |
   | -------------- | ------------------------- |
   | **Space name** | `heart-disease-predictor` |
   | **License**    | `MIT` (or any)            |
   | **SDK**        | **Gradio** ⬅️ IMPORTANT!  |
   | **Hardware**   | `CPU basic` (FREE)        |

5. Click **"Create Space"**

You'll see an empty space page with instructions.

---

## 📤 Step 4: Upload Files

### Files You Need

You need to upload **5 files** to your Space:

| File               | Where to Get It                                                |
| ------------------ | -------------------------------------------------------------- |
| `app.py`           | From `huggingface_deployment/` folder (I created this for you) |
| `requirements.txt` | From `huggingface_deployment/` folder                          |
| `README.md`        | From `huggingface_deployment/` folder                          |
| `model.pkl`        | From Step 1 (your notebook)                                    |
| `scaler.pkl`       | From Step 1 (your notebook)                                    |

### How to Upload

**Option A: Using the Web Interface (Easiest)**

1. On your Space page, click the **"Files"** tab

2. Click **"Add file"** → **"Upload files"**

3. Drag and drop all 5 files

4. Click **"Commit changes"**

**Option B: Using Git (Advanced)**

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/heart-disease-predictor
cd heart-disease-predictor

# Copy your files here
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/README.md .
cp /path/to/model.pkl .
cp /path/to/scaler.pkl .

# Push to Hugging Face
git add .
git commit -m "Add heart disease prediction app"
git push
```

---

## ⏳ Step 5: Wait for Build

After uploading your files:

1. Hugging Face will **automatically start building** your app

2. You'll see a **"Building"** status

3. Wait **2-5 minutes** for the build to complete

4. When done, you'll see your app running!

### What the App Looks Like

```
┌─────────────────────────────────────────────────────────────────┐
│  🫀 Heart Disease Risk Predictor                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Age: [__50__]     Sex: [Male ▼]     Chest Pain: [Type 0 ▼]    │
│                                                                  │
│  Blood Pressure: [__120__]    Cholesterol: [__200__]            │
│                                                                  │
│  ... (more fields) ...                                          │
│                                                                  │
│              [ 🔍 Submit ]                                       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  RESULT:                                                         │
│  💚 LOW RISK                                                     │
│  Probability: 23.5%                                              │
│  Keep up the healthy lifestyle!                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Step 6: Get Your API URL

### Your App URL

Your app is now live at:

```
https://huggingface.co/spaces/YOUR_USERNAME/heart-disease-predictor
```

### Your API URL

Gradio automatically creates an API endpoint:

```
https://YOUR_USERNAME-heart-disease-predictor.hf.space/api/predict
```

**Replace `YOUR_USERNAME` with your actual Hugging Face username!**

### Test the API

You can test it with curl:

```bash
curl -X POST "https://YOUR_USERNAME-heart-disease-predictor.hf.space/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}'
```

---

## 💻 Step 7: Use in Lovable App

### Add This Code to Your Lovable App

**Create a service file:**

```typescript
// services/heartPrediction.ts

// Replace with YOUR Hugging Face username
const API_URL =
  "https://YOUR_USERNAME-heart-disease-predictor.hf.space/api/predict";

interface PredictionInput {
  age: number;
  sex: number; // 0 = Female, 1 = Male
  cp: number; // Chest pain type (0-3)
  trestbps: number; // Resting blood pressure
  chol: number; // Cholesterol
  fbs: number; // Fasting blood sugar > 120 (0 or 1)
  restecg: number; // Resting ECG (0-2)
  thalach: number; // Max heart rate
  exang: number; // Exercise angina (0 or 1)
  oldpeak: number; // ST depression
  slope: number; // ST slope (0-2)
  ca: number; // Number of vessels (0-4)
  thal: number; // Thalassemia (0-3)
}

export async function predictHeartDisease(
  input: PredictionInput,
): Promise<string> {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: [
        input.age,
        input.sex,
        input.cp,
        input.trestbps,
        input.chol,
        input.fbs,
        input.restecg,
        input.thalach,
        input.exang,
        input.oldpeak,
        input.slope,
        input.ca,
        input.thal,
      ],
    }),
  });

  if (!response.ok) {
    throw new Error("Prediction failed");
  }

  const result = await response.json();
  return result.data; // This is the prediction result (markdown string)
}
```

### Use in a Component

```tsx
// components/HeartPredictionForm.tsx

import { useState } from "react";
import { predictHeartDisease } from "../services/heartPrediction";

export function HeartPredictionForm() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    age: 50,
    sex: 1,
    cp: 0,
    trestbps: 120,
    chol: 200,
    fbs: 0,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 0,
    slope: 0,
    ca: 0,
    thal: 1,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const prediction = await predictHeartDisease(formData);
      setResult(prediction);
    } catch (error) {
      console.error("Error:", error);
      setResult("Error making prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-center">
        🫀 Heart Disease Risk Assessment
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Age */}
        <div>
          <label className="block text-sm font-medium mb-1">Age</label>
          <input
            type="number"
            value={formData.age}
            onChange={(e) => setFormData({ ...formData, age: +e.target.value })}
            className="w-full p-2 border rounded"
            min="0"
            max="120"
          />
        </div>

        {/* Sex */}
        <div>
          <label className="block text-sm font-medium mb-1">Sex</label>
          <select
            value={formData.sex}
            onChange={(e) => setFormData({ ...formData, sex: +e.target.value })}
            className="w-full p-2 border rounded"
          >
            <option value={0}>Female</option>
            <option value={1}>Male</option>
          </select>
        </div>

        {/* Chest Pain Type */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Chest Pain Type
          </label>
          <select
            value={formData.cp}
            onChange={(e) => setFormData({ ...formData, cp: +e.target.value })}
            className="w-full p-2 border rounded"
          >
            <option value={0}>Typical Angina</option>
            <option value={1}>Atypical Angina</option>
            <option value={2}>Non-anginal Pain</option>
            <option value={3}>Asymptomatic</option>
          </select>
        </div>

        {/* Blood Pressure */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Resting Blood Pressure (mm Hg)
          </label>
          <input
            type="number"
            value={formData.trestbps}
            onChange={(e) =>
              setFormData({ ...formData, trestbps: +e.target.value })
            }
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Cholesterol */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Cholesterol (mg/dl)
          </label>
          <input
            type="number"
            value={formData.chol}
            onChange={(e) =>
              setFormData({ ...formData, chol: +e.target.value })
            }
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Add more fields as needed... */}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "🔄 Analyzing..." : "🔍 Predict Heart Risk"}
        </button>
      </form>

      {/* Result Display */}
      {result && (
        <div className="mt-6 p-4 rounded-lg bg-gray-50 border">
          <div dangerouslySetInnerHTML={{ __html: result }} />
        </div>
      )}
    </div>
  );
}
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### ❌ Build Failed

**Cause:** Missing dependencies or wrong file names

**Solution:**

1. Check that `requirements.txt` has all packages
2. Make sure `app.py` is named exactly `app.py`
3. Check the build logs for specific errors

#### ❌ Model Not Loading

**Cause:** scikit-learn version mismatch

**Solution:** Make sure `requirements.txt` has the same version as your training:

```
scikit-learn==1.3.2
```

#### ❌ CORS Error in Browser

**Cause:** Gradio usually handles CORS, but sometimes browsers block

**Solution:** The Gradio API should work. If not, try:

```typescript
// Add mode: 'cors' to your fetch
const response = await fetch(API_URL, {
  method: "POST",
  mode: "cors",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ data: [...] })
});
```

#### ❌ API Returns Error

**Cause:** Wrong data format

**Solution:** Make sure you send data as an array in the correct order:

```javascript
{
  "data": [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
}
```

#### ❌ Predictions Are Wrong

**Cause:** Features in wrong order or scaler not used

**Solution:**

1. Features MUST be in this exact order:
   `[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]`
2. Make sure you uploaded `scaler.pkl`

---

## 📁 Files Reference

### Files in `huggingface_deployment/` folder:

| File                 | Purpose                      |
| -------------------- | ---------------------------- |
| `app.py`             | Main Gradio application      |
| `requirements.txt`   | Python dependencies          |
| `README.md`          | Space description            |
| `ADD_TO_NOTEBOOK.py` | Code to add to your notebook |

### File Contents

#### app.py

```python
# Main Gradio application that:
# 1. Loads model.pkl and scaler.pkl
# 2. Creates a web interface with 13 input fields
# 3. Returns prediction, probability, and risk message
```

#### requirements.txt

```
gradio==4.44.0
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.3
```

---

## ✅ Checklist

Before deploying, make sure you have:

- [ ] Trained your model in the notebook
- [ ] Run the export code (Step 1)
- [ ] Downloaded `model.pkl` and `scaler.pkl`
- [ ] Created Hugging Face account
- [ ] Created a new Space with **Gradio** SDK
- [ ] Uploaded all 5 files:
  - [ ] `app.py`
  - [ ] `requirements.txt`
  - [ ] `README.md`
  - [ ] `model.pkl`
  - [ ] `scaler.pkl`
- [ ] Waited for build to complete
- [ ] Tested the app works
- [ ] Copied the API URL for your Lovable app

---

## 🎉 Done!

Congratulations! Your ML model is now deployed and accessible via API without any server management!

**Your URLs:**

- **Web App:** `https://huggingface.co/spaces/YOUR_USERNAME/heart-disease-predictor`
- **API:** `https://YOUR_USERNAME-heart-disease-predictor.hf.space/api/predict`

---

_Need help? Check the [Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces)_
