import nlpaug.augmenter.word as naw


aug = naw.ContextualWordEmbsAug(model_type="bert", model_path="bert-base-uncased")

text = "I love you"

augmented_data = aug.augment(text)
print(augmented_data)