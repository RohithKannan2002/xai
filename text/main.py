"""Complete text XAI practical from the uploaded notebook."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, InputXGradient
from lime.lime_text import LimeTextExplainer

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device); model.eval()
text = "The movie was wonderful, exciting and enjoyable."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
with torch.no_grad():
    logits = model(**inputs).logits; probabilities = torch.softmax(logits, dim=1)[0]
predicted_class = int(probabilities.argmax().item()); class_names = ["NEGATIVE", "POSITIVE"]
print("Text:", text); print("Predicted sentiment:", class_names[predicted_class]); print("Probabilities:", probabilities.cpu().numpy())

# 1. Tokenization
tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
print("\nTokens:", tokens)
plt.figure(figsize=(10, 2)); plt.axis("off")
for i, token in enumerate(tokens): plt.text(0.02 + i * 0.12, 0.5, token, fontsize=10, bbox=dict(boxstyle="round", alpha=0.2))
plt.title("Tokenization"); plt.tight_layout(); plt.show()

# 2. Word embeddings
embedding_layer = model.get_input_embeddings()
with torch.no_grad(): embeddings = embedding_layer(input_ids)
print("\nEmbedding shape:", tuple(embeddings.shape))
embedding_values = embeddings[0, :min(8, len(tokens)), :].detach().cpu().numpy()
plt.figure(figsize=(10, 4)); plt.imshow(embedding_values, aspect="auto", cmap="viridis")
plt.xlabel("Embedding dimensions"); plt.ylabel("Tokens"); plt.yticks(range(min(8, len(tokens))), tokens[:8])
plt.title("Word Embeddings"); plt.colorbar(label="Embedding value"); plt.tight_layout(); plt.show()

# 3. LIME for text
def predict_proba(texts):
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad(): return torch.softmax(model(**batch).logits, dim=1).cpu().numpy()
lime_explainer = LimeTextExplainer(class_names=class_names)
lime_exp = lime_explainer.explain_instance(text, predict_proba, num_features=8, num_samples=500)
lime_values = lime_exp.as_list(label=predicted_class)
print("\nLIME explanation:")
for word, score in lime_values: print(word, round(score, 4))
words = [item[0] for item in lime_values]; scores = [item[1] for item in lime_values]
plt.figure(figsize=(8, 4)); plt.barh(words[::-1], scores[::-1]); plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Local contribution"); plt.title("LIME - Text Explanation"); plt.tight_layout(); plt.show()

# Shared forward function for attribution
def forward_from_embeddings(embeds, mask):
    return model(inputs_embeds=embeds, attention_mask=mask).logits

# 4. Gradient x Input
word_embeddings = embedding_layer(input_ids).detach(); word_embeddings.requires_grad_(True)
gradient_x_input = InputXGradient(forward_from_embeddings)
gxi_attr = gradient_x_input.attribute(word_embeddings, additional_forward_args=(attention_mask,), target=predicted_class)
gxi_scores = gxi_attr.sum(dim=-1)[0].detach().cpu().numpy()
print("\nGradient x Input scores:")
for token, score in zip(tokens, gxi_scores): print(token, round(float(score), 4))
plt.figure(figsize=(9, 4)); plt.bar(range(len(tokens)), gxi_scores); plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("Attribution"); plt.title("Gradient x Input"); plt.tight_layout(); plt.show()

# 5. Layer Integrated Gradients
# Explain the output of the first Transformer layer.
target_layer = model.distilbert.transformer.layer[0]
lig = LayerIntegratedGradients(forward_from_embeddings, target_layer)
baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
input_embeddings = embedding_layer(input_ids); baseline_embeddings = embedding_layer(baseline_ids)
lig_attr = lig.attribute(inputs=input_embeddings, baselines=baseline_embeddings, additional_forward_args=(attention_mask,), target=predicted_class, n_steps=20)
lig_scores = lig_attr.sum(dim=-1)[0].detach().cpu().numpy()
print("\nLayer Integrated Gradients scores:")
for token, score in zip(tokens, lig_scores): print(token, round(float(score), 4))
plt.figure(figsize=(9, 4)); plt.bar(range(len(tokens)), lig_scores); plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("Attribution"); plt.title("Layer Integrated Gradients"); plt.tight_layout(); plt.show()
print("\nAll five text XAI demonstrations completed!")
