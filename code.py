# Install required packages
!pip install datasets
!pip install scikit-learn
!pip install ibm-watson-machine-learning==1.0.312

# Import OS and data loading libraries
import os, getpass
from pandas import read_csv

# Watsonx API connection
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("Please enter your WML api key (hit enter): ")
}

#  Defining the project ID
try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")

# Display project ID
project_id

# Creating Legal Document Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    "Document_Text": [
        "This Non-Disclosure Agreement is made between Company A and Company B...",
        "This Service Agreement outlines the responsibilities of the provider and the client...",
        "This Employment Contract sets forth the terms of employment between the employer and employee...",
        "This Lease Agreement is entered into between the landlord and tenant...",
        "This Consulting Agreement is made effective as of the date between the consultant and the client...",
        "This Purchase Order confirms the agreement to purchase goods from the vendor...",
        "This Licensing Agreement grants the licensee certain rights to use intellectual property...",
        "This Partnership Agreement details the terms of collaboration between the two firms...",
        "This Memorandum of Understanding outlines the preliminary terms agreed by both parties...",
        "This Loan Agreement specifies the obligations of the borrower and the lender..."
    ],
    "Document_Type": [
        "NDA", "Service Agreement", "Employment Contract", "Lease Agreement", "Consulting Agreement",
        "Purchase Order", "Licensing Agreement", "Partnership Agreement", "MOU", "Loan Agreement"
    ]
}

df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df.to_csv("legal_train.csv", index=False)
test_df.to_csv("legal_test.csv", index=False)

# Load Training Data
train_data = pd.read_csv("legal_train.csv")
train_data.head(5)

# Load Test Data
test_data = pd.read_csv("legal_test.csv")
test_data.head(5)

# Shape of training data
train_data.shape

# Shape of test data
test_data.shape

#  Import FLAN model ID
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
model_id = ModelTypes.FLAN_UL2

#  Prompt for classification
classification_instruction = """
Classify the legal document into one of the following types:
'NDA', 'Service Agreement', 'Employment Contract', 'Lease Agreement', 'Consulting Agreement',
'Purchase Order', 'Licensing Agreement', 'Partnership Agreement', 'MOU', 'Loan Agreement'.

Document: This Non-Disclosure Agreement is made between Company A and Company B...
Type: NDA\n\n
"""

#  Define generation parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

parameters = {
    GenParams.MAX_NEW_TOKENS: 10
}

#  Load FLAN model using WML
from ibm_watson_machine_learning.foundation_models import Model

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

#  Run classification on test documents
results = []
documents = list(test_data.Document_Text)

for doc in documents:
    results.append(model.generate_text(prompt=" ".join([classification_instruction, doc])))

#Print documents
documents

# Print classification results
results
