
# Program 20: Information Extraction from Structured and Unstructured Documents

import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Unstructured document
text = """
Gauri Kulshrestha works at Microsoft.
She lives in New Delhi and joined the company in 2025.
Contact email: gauri@example.com.
"""

print("========== UNSTRUCTURED DOCUMENT ==========")
print(text)

# Extract named entities using spaCy
doc = nlp(text)

print("\n========== NAMED ENTITY EXTRACTION ==========")

for entity in doc.ents:
    print(entity.text, "->", entity.label_)

# Extract email using regular expression
email_pattern = r'[\w.-]+@[\w.-]+\.\w+'
emails = re.findall(email_pattern, text)

print("\n========== EMAIL EXTRACTION ==========")
for email in emails:
    print("Email:", email)

# Structured document represented as a dictionary
employee = {
    "name": "Gauri Kulshrestha",
    "company": "Microsoft",
    "city": "New Delhi",
    "year_joined": 2025
}

print("\n========== STRUCTURED DATA EXTRACTION ==========")

for key, value in employee.items():
    print(key, ":", value)
