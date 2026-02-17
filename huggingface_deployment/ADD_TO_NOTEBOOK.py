# =====================================================
# ADD THIS CODE AT THE END OF YOUR JUPYTER NOTEBOOK
# Run this AFTER you have trained your model
# =====================================================

import pickle

# ---------------------------------------------------
# STEP 1: Make sure you have the scaler and model
# ---------------------------------------------------
# If you ran all the cells above, you should already have:
# - scaler (StandardScaler)
# - rf_model (RandomForestClassifier)

# If you get an error, run these cells first:
# - The cell with: scaler = StandardScaler()
# - The cell with: rf_model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# STEP 2: Save the model and scaler
# ---------------------------------------------------
print("Saving model and scaler...")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: scaler.pkl")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Saved: model.pkl")

# ---------------------------------------------------
# STEP 3: Test that the files work
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
# STEP 4: Download instructions
# ---------------------------------------------------
print("\n" + "="*50)
print("📥 NEXT STEPS:")
print("="*50)
print("""
1. DOWNLOAD these 2 files from your working directory:
   - model.pkl
   - scaler.pkl

2. If using Google Colab, run this to download:
   
   from google.colab import files
   files.download('model.pkl')
   files.download('scaler.pkl')

3. Upload these files to Hugging Face Spaces!
""")
