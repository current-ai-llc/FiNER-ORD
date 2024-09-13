import modal
import torch
from pydantic import BaseModel, Field
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

# Define the Modal stub
app = modal.App("ner-classifier")

# Create a Modal image with the required dependencies
inference_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "scikit-learn", "accelerate"
)

# Define the labels
labels = ["B-LOC", "I-LOC", "O", "B-ORG", "I-ORG", "B-PER", "I-PER"]


class NERResult(BaseModel):
    tokens: list[str] = Field(..., description="List of tokens in the input text")
    labels: list[str] = Field(..., description="Predicted NER labels for each token")


@app.function(
    name="classify_ner",
    image=inference_image,
    gpu="any",
    mounts=[modal.Mount.from_local_dir(".", remote_path="/code")],
    timeout=300,
)
def classify_ner(texts: list[str]) -> list[NERResult]:
    import time

    start_time = time.time()

    # Load tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-large", add_prefix_space=True
    )
    model = RobertaForTokenClassification.from_pretrained("roberta-large", num_labels=7)

    # Load the best model weights
    checkpoint = torch.load("/code/best_model.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    for text in texts:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted labels
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_labels = [labels[p] for p in predictions[0]]

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Remove special tokens and their corresponding labels
        valid_tokens = []
        valid_labels = []
        for token, label in zip(tokens, predicted_labels):
            if token not in [
                tokenizer.cls_token,
                tokenizer.sep_token,
                tokenizer.pad_token,
            ]:
                valid_tokens.append(token)
                valid_labels.append(label)

        results.append(NERResult(tokens=valid_tokens, labels=valid_labels))

    end_time = time.time()
    print(f"Compute Time taken: {end_time - start_time} seconds")
    return results


@app.local_entrypoint()
def main():
    # Test the function locally
    texts = [
        "John Smith works at Microsoft in Seattle.",
        "The Eiffel Tower is in Paris, France.",
    ]
    results = classify_ner.remote(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print("Tokens:", result["tokens"])
        print("Labels:", result["labels"])
        print()


if __name__ == "__main__":
    main()
