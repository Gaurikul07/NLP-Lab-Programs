import re
import spacy
nlp = spacy.load("en_core_web_sm")
text = """
Gauri Kulshrestha works at Microsoft.
She lives in New Delhi and joined the company in 2025.
Contact email: gauri@example.com.
"""

print("========== UNSTRUCTURED DOCUMENT ==========")
print(text)
doc = nlp(text)

print("\n========== NAMED ENTITY EXTRACTION ==========")

for entity in doc.ents:
    print(entity.text, "->", entity.label_)
email_pattern = r'[\w.-]+@[\w.-]+\.\w+'
emails = re.findall(email_pattern, text)

print("\n========== EMAIL EXTRACTION ==========")
for email in emails:
    print("Email:", email)
employee = {
    "name": "Gauri Kulshrestha",
    "company": "Microsoft",
    "city": "New Delhi",
    "year_joined": 2025
}

print("\n========== STRUCTURED DATA EXTRACTION ==========")

for key, value in employee.items():
    print(key, ":", value)
