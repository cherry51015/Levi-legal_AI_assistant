import faiss
import json
import numpy as np


# Step 1: Load embeddings.jsonl

embeddings = []
ids = []
texts = []

with open("embeddings.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ids.append(obj["id"])
        texts.append(obj.get("text", ""))  # store original text if available
        embeddings.append(obj["embedding"])

embeddings = np.array(embeddings, dtype="float32")
print(f"✅ Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

# Step 2: Create FAISS index

dimension = embeddings.shape[1]

# Normalize embeddings for cosine similarity (better for semantic search)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors")


# Step 3: Save FAISS index + Metadata

faiss.write_index(index, "faiss_index.bin")

metadata = {
    "ids": ids,
    "texts": texts
}

with open("faiss_index.bin.meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("💾 Saved index as faiss_index.bin and metadata as faiss_index.bin.meta.json")


index = faiss.read_index("faiss_index.bin")

query = embeddings[0].reshape(1, -1)

k = 5
scores, neighbors = index.search(query, k)

print("\n🔎 Example Query:")
print("Nearest neighbor indices:", neighbors[0])
print("Scores:", scores[0])

with open("faiss_index.bin.meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

neighbor_ids = [meta["ids"][i] for i in neighbors[0]]
neighbor_texts = [meta["texts"][i] for i in neighbors[0]]

print("Neighbor IDs:", neighbor_ids)
print("Neighbor Texts:", neighbor_texts)
