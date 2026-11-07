"""Local Granite embeddings: first run downloads model weights from Hugging Face."""
from sentence_transformers import SentenceTransformer, util

MODEL_ID = "ibm-granite/granite-embedding-english-r2"

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_ID)
    descriptions = [
        "An unauthenticated remote attacker can execute code on a VPN gateway.",
        "An authentication bypass allows remote access to a VPN appliance.",
        "A local user can read files without sufficient permissions.",
    ]
    vectors = model.encode(descriptions, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(vectors[0:1], vectors)[0]
    print("Embedding shape:", tuple(vectors.shape))
    for idx, score in enumerate(scores):
        print(f"{score.item():.3f} | {descriptions[idx]}")
