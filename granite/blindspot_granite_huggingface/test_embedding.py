from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "ibm-granite/granite-embedding-97m-multilingual-r2"
)

descriptions = [
    "A vulnerability allows unauthorized access to a web application.",
    "An authentication bypass permits attackers to access restricted resources.",
    "A software bug causes an application to crash."
]

embeddings = model.encode(descriptions)

print("Embedding shape:", embeddings.shape)

similarities = model.similarity(embeddings, embeddings)

print(similarities)