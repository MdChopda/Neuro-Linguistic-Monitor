import pandas as pd
import torch
import math
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.tokenize import word_tokenize

# --- 1. SETUP & SAFETY CHECKS ---
print("⏳ Initializing Research Environment...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load Model (Cached)
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# --- 2. ANALYSIS ENGINE ---
def analyze_patient(text):
    # Calculate Perplexity (Confusion)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    ppl = math.exp(outputs.loss.item())

    # Calculate Richness (TTR)
    tokens = word_tokenize(text.lower())
    ttr = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
    
    return ppl, ttr

# --- 3. BATCH PROCESSING ---
try:
    # Load Data
    print("📂 Loading 'large_dataset.csv'...")
    df = pd.read_csv("large_dataset.csv")
    print(f"✅ Successfully loaded {len(df)} patient records.")

    results = []
    print("🚀 Running Neuro-Linguistic Analysis (This may take a minute)...")
    
    for index, row in df.iterrows():
        text = str(row['text'])
        label = row['label']
        
        try:
            ppl, ttr = analyze_patient(text)
            
            # DIAGNOSTIC LOGIC (The "AI Doctor")
            # If vocabulary is low (< 0.55) AND confusion is high (> 15), flag as Sick
            predicted_label = "alzheimers" if (ttr < 0.55 and ppl > 15) else "healthy"
            
            results.append({
                "Actual": label,
                "Predicted": predicted_label,
                "Perplexity": ppl,
                "TTR": ttr
            })
        except Exception as e:
            pass # Skip bad rows

    # Convert to DataFrame
    res_df = pd.DataFrame(results)

    # --- 4. VISUALIZATION & STATISTICS ---
    print("\n" + "="*40)
    print(f"📊 FINAL RESULTS (n={len(res_df)})")
    print("="*40)

    # Calculate Accuracy
    acc = accuracy_score(res_df['Actual'], res_df['Predicted'])
    print(f"🏆 Model Accuracy: {acc*100:.1f}%")

    # Generate Confusion Matrix
    cm = confusion_matrix(res_df['Actual'], res_df['Predicted'], labels=["healthy", "alzheimers"])
    print("\n--- Confusion Matrix ---")
    print(f"True Healthy:      {cm[0][0]}  (Correct)")
    print(f"False Alarm:       {cm[0][1]}  (Error)")
    print(f"Missed Diagnosis:  {cm[1][0]}  (Error)")
    print(f"True Alzheimer's:  {cm[1][1]}  (Correct)")

    # --- PLOT: Professional Clinical Cluster Map ---
    plt.figure(figsize=(12, 7))
    
    # Create 'Status' column for cleaner Legend
    def get_status(row):
        if row['Actual'] == 'healthy' and row['Predicted'] == 'healthy':
            return 'Healthy (Correct)'
        elif row['Actual'] == 'alzheimers' and row['Predicted'] == 'alzheimers':
            return 'Alzheimers (Correct)'
        elif row['Actual'] == 'healthy' and row['Predicted'] == 'alzheimers':
            return 'False Alarm (Error)'
        else:
            return 'Missed Diagnosis (Error)'

    res_df['Status'] = res_df.apply(get_status, axis=1)

    # Define professional medical colors
    custom_palette = {
        'Healthy (Correct)': '#2ca02c',      # Green
        'Alzheimers (Correct)': '#d62728',   # Red
        'False Alarm (Error)': '#ff7f0e',    # Orange
        'Missed Diagnosis (Error)': '#000000' # Black
    }

    # Draw the Scatter Plot
    sns.scatterplot(
        data=res_df, 
        x='TTR', 
        y='Perplexity', 
        hue='Status', 
        style='Status',
        palette=custom_palette,
        s=120, # Bigger dots
        alpha=0.8 # Slight transparency
    )

    # Add the "Safety Lines"
    plt.axvline(x=0.55, color='gray', linestyle='--', linewidth=1.5, label='Vocab Threshold')
    plt.axhline(y=15, color='gray', linestyle=':', linewidth=1.5, label='Confusion Threshold')

    # Clean up the labels
    plt.title(f'Neuro-Linguistic Diagnostic Clusters (n={len(res_df)})', fontsize=14, fontweight='bold')
    plt.xlabel('Vocabulary Richness (TTR) -> Higher is Better', fontsize=11)
    plt.ylabel('Syntactic Confusion (Perplexity) -> Lower is Better', fontsize=11)
    
    # Fix the Legend box location
    plt.legend(title='Diagnostic Result', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # Save High Res
    plt.savefig('clinical_study_results.png', dpi=300)
    print("\n✅ Graph saved as 'clinical_study_results.png'")
    print("✅ Analysis complete.")
    plt.show()

except FileNotFoundError:
    print("❌ Error: Please create 'large_dataset.csv' on your Desktop first.")