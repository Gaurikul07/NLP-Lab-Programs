
# Program 7: Coreference Resolution using Coreferee

import spacy
import coreferee

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Add Coreferee to spaCy pipeline
nlp.add_pipe("coreferee")

# Input text
text = """
Gauri is a student.
She is learning Natural Language Processing.
Gauri enjoys NLP because it helps her understand language.
"""

# Process text
doc = nlp(text)

# Display original text
print("Original Text:")
print(text)

# Display coreference clusters
print("\nCoreference Clusters:")

for chain in doc._.coref_chains:
    print(chain)

# Display resolved text
print("\nResolved Text:")

resolved_text = coreferee.resolve(doc)
print(resolved_text)
