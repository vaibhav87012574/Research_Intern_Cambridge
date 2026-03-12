# Research_Intern_Cambridge
# Motivation & Engagement Analysis System

## Overview

This repository contains tools for analyzing **motivation and engagement in conversations using modern NLP models**.

The project includes:

1. **Motivation Classification**
   - Classifies text as **Intrinsic** or **Extrinsic motivation**.
   - Uses a fine-tuned transformer model.
   - Includes an interactive **Streamlit web app**.

2. **Batch Motivation Classification**
   - GPU-accelerated classification of large datasets.
   - Uses **facebook/bart-large-mnli** for zero-shot classification.

3. **Multilingual Engagement Analysis**
   - Advanced conversation analysis system.
   - Supports multiple languages.
   - Extracts engagement features and generates engagement scores.

---

# Project Structure


.
├── app.py
├── engagement_classifier.py
├── my_script.py
├── model/
├── notebooks/
│ ├── Classification.ipynb
│ ├── finetune.ipynb
│ ├── motivation_classifier.ipynb
│ ├── venture_challenge_classifier.ipynb
├── README.md


---

# Features

## 1. Motivation Analyzer (Streamlit App)

A web application that predicts whether a sentence reflects:

### Intrinsic Motivation
- Curiosity
- Passion
- Personal growth

### Extrinsic Motivation
- Grades
- Money
- Rewards
- Avoiding punishment

Example:

Input
```
I want to learn this topic because it fascinates me.
```

Output
```
Prediction: Intrinsic Motivation
Confidence: 92%
```

---

# 2. Batch Motivation Classification (GPU)

The script performs **large-scale motivation classification on CSV datasets**.

### Model Used

```
facebook/bart-large-mnli
```

### Workflow

1. Load CSV data
2. Extract mentor motivation text
3. Run zero-shot classification
4. Save results to CSV

### Input Columns

```
person1_motivation_mentorship
person2_motivation_mentorship
person1_is_mentor
```

### Output Columns

```
mentor_motivation_str
motivation
confidence
```

---

# 3. Multilingual Engagement Analysis System

This module performs **advanced engagement analysis across conversations**.

### Supported Languages

- English
- Spanish
- Russian
- Arabic
- Indonesian

### Extracted Engagement Features

#### Message Interaction
- Number of messages
- Questions
- Agreements / disagreements
- Conflicts

#### Communication Style
- Curiosity
- Proactiveness
- Politeness
- Formality
- Persistence

#### Linguistic Metrics
- Lexical diversity
- Text complexity
- Concreteness

#### Business Context
- Customer focus
- Market focus
- Product discussion
- Action orientation

---

# Engagement Output

The system generates:

### Engagement Index

```
0 → Low engagement
1 → High engagement
```

### Engagement Dimensions

- Cross-cultural interaction
- Information exchange quality
- Action orientation

### Insights

- Language performance
- Communication patterns
- Multilingual advantages

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/motivation-analysis.git

cd motivation-analysis
2. Install Dependencies
pip install -r requirements.txt

Typical dependencies:

torch
transformers
streamlit
pandas
numpy
scikit-learn
tqdm
langdetect
textblob
nltk
sentence-transformers
Running the Streamlit App
streamlit run app.py

Then open:

http://localhost:8501
Running Batch Motivation Classification
python my_script.py

Input file:

merged_conversations_with_translations.csv

Output file:

final_result.csv
Running Multilingual Engagement Analysis

Example:

from engagement_classifier import AdvancedMultilingualEngagementSystem

system = AdvancedMultilingualEngagementSystem()

results = system.analyze_multilingual_conversations(
    "conversation_data.csv"
)

Output includes:

engagement_outcomes
features
validation_results
insights
