from transformers.utils import cached_file

model_path = cached_file("sentence-transformers/all-MiniLM-L6-v2", "config.json")
print(model_path)