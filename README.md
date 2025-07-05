# 🧾 Legal Document Classifier using IBM Watsonx

This project uses IBM Watsonx's FLAN-UL2 foundation model to classify legal documents into categories such as NDA, Employment Contract, Service Agreement, and more. It leverages prompt-based zero-shot/few-shot learning and requires no manual feature engineering.

---

## 🚀 Features

- Uses FLAN-UL2 LLM hosted on IBM Watsonx
- Classifies legal documents using natural language prompts
- Zero training needed – prompt-based inference
- Easily customizable for more document types

---

## 📂 Dataset

A small sample dataset of 10 legal document types is included for demonstration:

- NDA
- Service Agreement
- Employment Contract
- Lease Agreement
- Consulting Agreement
- Purchase Order
- Licensing Agreement
- Partnership Agreement
- MOU
- Loan Agreement

---

## 🧠 Model

**Model Used:** `FLAN-UL2` via `ibm-watson-machine-learning` SDK  
**Text Generation Parameters:**  
- `max_new_tokens = 10`

---

## 🛠️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/legal-doc-classifier.git
   cd legal-doc-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add your IBM Watsonx API credentials in the notebook when prompted.

📊 Sample Prompt
text
Copy
Edit
Classify the legal document into one of the following types:
'NDA', 'Service Agreement', 'Employment Contract', 'Lease Agreement', 'Consulting Agreement',
'Purchase Order', 'Licensing Agreement', 'Partnership Agreement', 'MOU', 'Loan Agreement'.

Document: This Non-Disclosure Agreement is made between Company A and Company B...
Type:
Output
Model generates the predicted document type based on the given text.
