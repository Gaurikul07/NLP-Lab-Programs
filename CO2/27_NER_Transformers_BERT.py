from transformers import pipeline
ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)
text = "Gauri works at Microsoft in New Delhi, India."

print("========== GIVEN TEXT ==========")
print(text)
entities = ner(text)

print("\n========== NAMED ENTITIES ==========")

if entities:
    for entity in entities:
        print("Entity:", entity["word"])
        print("Type:", entity["entity_group"])
        print("Confidence:", round(entity["score"], 4))
        print()
else:
    print("No named entities found.")
