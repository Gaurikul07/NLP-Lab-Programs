!pip install -q spacy coreferee
!python -m spacy download en_core_web_sm
!python -m coreferee install en
import spacy
import coreferee
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("coreferee")
text = """
Gauri is a student.
She is learning Natural Language Processing.
Gauri enjoys NLP because it helps her understand language.
"""
doc = nlp(text)
print("Original Text:")
print(text)
print("\nCoreference Clusters:")

for chain in doc._.coref_chains:
    print(chain)
print("\nResolved Text:")

resolved_text = coreferee.resolve(doc)
print(resolved_text)
