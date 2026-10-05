import spacy
nlp = spacy.load("en_core_web_sm")
text = "Elon Musk founded SpaceX in the United States. He also leads Tesla in California."

doc = nlp(text)

print("========== GIVEN TEXT ==========")
print(text)

print("\n========== NAMED ENTITY RECOGNITION ==========")

for entity in doc.ents:
    print(entity.text, "->", entity.label_)

print("\n========== ENTITY DETAILS ==========")

for entity in doc.ents:
    print("Entity:", entity.text)
    print("Label:", entity.label_)
    print("Description:", spacy.explain(entity.label_))
    print()
