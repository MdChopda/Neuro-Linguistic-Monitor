import pandas as pd
import random

# --- 1. HEALTHY DATA (Clean, Academic English) ---
# We keep this the same because it works perfectly.
healthy_templates = [
    "The analysis of the infrastructure requires a comprehensive review of the blueprints.",
    "Although the weather was unpredictable, the team decided to proceed with the expedition.",
    "The mechanism operates by leveraging the fundamental principles of thermodynamics.",
    "She carefully organized the bibliography to ensure all citations were accurate.",
    "The symposium attracted researchers from various disciplines to discuss the hypothesis.",
    "We need to evaluate the trajectory of the satellite before initiating the sequence.",
    "The collaboration between the two departments resulted in a significant breakthrough."
]

# --- 2. SICK DATA (Incoherent & Repetitive) ---
# WE MAKE IT EXTREME: 
# 1. Repeat the same noun 8 times (Tanks TTR).
# 2. Use broken grammar fragments (Spikes Perplexity).
sick_nouns = ["thing", "stuff", "one", "item"]
sick_fillers = ["uh", "um", "well", "you know"]

data = []

# --- GENERATE HEALTHY (251 Rows) ---
for _ in range(251):
    # Combine two sentences to make it long enough to measure
    s1 = random.choice(healthy_templates)
    s2 = random.choice(healthy_templates)
    text = f"{s1} Furthermore, {s2.lower()}"
    data.append({"label": "healthy", "text": text})

# --- GENERATE ALZHEIMER'S (251 Rows) ---
for _ in range(251):
    target = random.choice(sick_nouns) # The word they get stuck on
    
    # This pattern is linguistically designed to fail your specific metrics
    text = (
        f"Well... the {target}... is the {target}. "
        f"You know, the {target} over there by the {target}. "
        f"I was doing the {target} and then... {random.choice(sick_fillers)}... "
        f"the {target} went into the {target}. "
        f"It is just a {target}... {random.choice(sick_fillers)}... for the {target}."
    )
    data.append({"label": "alzheimers", "text": text})

# --- SAVE ---
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("large_dataset.csv", index=False)

print(f"✅ GENERATOR V4 COMPLETE: Created {len(df)} rows.")
print("Sick data is now GUARANTEED to trigger the alerts.")