import modal
import torch
from pydantic import BaseModel, Field
from transformers import BertForSequenceClassification, BertTokenizer

# Define the Modal stub
app = modal.App("calendar-events-classifier")

# Create a Modal image with the required dependencies
inference_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "scikit-learn", "accelerate"
)

# Define the labels
labels = [
    "EARNINGS_UPDATE",
    "COMPANY_EVENT_UPDATE",
    "COMPANY_STRATEGY_UPDATE",
    "DATA_CHECKS_READACROSS_UPDATE",
    "STOCK_OPINION_UPDATE",
    "SELL_SIDE_EVENT_NOTIFICATION",
]

model_path = "."


class CalendarClassificationResult(BaseModel):
    scores: dict[str, float] = Field(
        ...,
        description="A dictionary mapping each label to its probability",
        example={
            "EARNINGS": 0.9501314759254456,
            "INVESTOR_UPDATE": 0.008619148284196854,
            "INVESTOR_CONF": 0.006564429495483637,
            "INDUSTRY_CONF": 0.0061136940494179726,
            "COMPANY_EVENT": 0.0061864168383181095,
            "ANALYST_DAY": 0.006823734380304813,
            "SHAREHOLDER_MEETING": 0.006913966964930296,
        },
    )
    selected_labels: list[str] = Field(
        ...,
        description="A list of labels that were selected based on a threshold",
        example=["EARNINGS"],
    )


class NoteClassificationResult(BaseModel):
    scores: dict[str, float] = Field(
        ...,
        description="A dictionary mapping each label to its probability",
        example={
            "EARNINGS_UPDATE": 0.9501314759254456,
            "COMPANY_EVENT_UPDATE": 0.008619148284196854,
            "COMPANY_STRATEGY_UPDATE": 0.006564429495483637,
            "DATA_CHECKS_READACROSS_UPDATE": 0.0061136940494179726,
            "STOCK_OPINION_UPDATE": 0.0061864168383181095,
            "SELL_SIDE_EVENT_NOTIFICATION": 0.006823734380304813,
        },
    )
    selected_labels: list[str] = Field(
        ...,
        description="A list of labels that were selected based on a threshold",
        example=["EARNINGS_UPDATE"],
    )


@app.function(
    name="classify_sellside_note",
    image=inference_image,
    gpu=modal.gpu.L4(count=1),
    volumes={
        "/model": modal.Volume.from_name("ml-tasks-volume", create_if_missing=True)
    },
    timeout=300,
)
def classify_sellside_note(texts) -> list[NoteClassificationResult]:
    import time

    start_time = time.time()
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "/model/sell_side_document_category_transformer_model"
    )

    # Move model to appropriate device (CPU, GPU, or MPS)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)

    results = []
    for text in texts:
        # Prepare input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            device
        )

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()

        # Define your labels
        labels = [
            "EARNINGS_UPDATE",
            "COMPANY_EVENT_UPDATE",
            "COMPANY_STRATEGY_UPDATE",
            "DATA_CHECKS_READACROSS_UPDATE",
            "STOCK_OPINION_UPDATE",
            "SELL_SIDE_EVENT_NOTIFICATION",
        ]

        # Create a dictionary of label probabilities
        scores_dict = {label: score for label, score in zip(labels, probabilities)}

        # Select labels above a certain threshold (e.g., 0.5)
        selected_labels = [label for label, score in scores_dict.items() if score > 0.5]

        # print("Probabilities:", scores_dict)
        # print("Selected labels:", selected_labels)
        results.append(
            dict(
                CalendarClassificationResult(
                    scores=scores_dict,
                    selected_labels=selected_labels,
                )
            )
        )
    end_time = time.time()
    print(f"Compute Time taken: {end_time - start_time} seconds")
    return results


@app.function(
    name="classify_calendar_event",
    image=inference_image,
    gpu=modal.gpu.L4(count=1),
    volumes={
        "/model": modal.Volume.from_name("ml-tasks-volume", create_if_missing=True)
    },
    timeout=300,
)
def classify_calendar_event(texts) -> list[CalendarClassificationResult]:
    import time

    start_time = time.time()
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "/model/cal_event_category_transformer_model"
    )

    # Move model to appropriate device (CPU, GPU, or MPS)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)

    results = []
    for text in texts:
        # Prepare input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            device
        )

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()

        # Define your labels
        labels = [
            "EARNINGS",
            "INVESTOR_UPDATE",
            "INVESTOR_CONF",
            "INDUSTRY_CONF",
            "COMPANY_EVENT",
            "ANALYST_DAY",
            "SHAREHOLDER_MEETING",
        ]

        # Create a dictionary of label probabilities
        scores_dict = {label: score for label, score in zip(labels, probabilities)}

        # Select labels above a certain threshold (e.g., 0.5)
        selected_labels = [label for label, score in scores_dict.items() if score > 0.5]

        # print("Probabilities:", scores_dict)
        # print("Selected labels:", selected_labels)
        results.append(
            dict(
                CalendarClassificationResult(
                    scores=scores_dict,
                    selected_labels=selected_labels,
                )
            )
        )
    end_time = time.time()
    print(f"Compute Time taken: {end_time - start_time} seconds")
    return results


# @app.local_entrypoint()
# def main():
#     import time

#     start_time = time.time()
#     print(
#         len(
#             classify_calendar_event.remote(
#                 [
#                     "Goldman Sachs Seventh Annual Leveraged Finance and Credit Conference \u2013 Rancho Palos Verdes, CA"
#                 ]
#                 * 2000
#             )
#         )
#     )
#     end_time = time.time()
#     print(f"Total Time taken: {end_time - start_time} seconds")
#     classify = modal.Function.from_name(
#         app_name="calendar-events-classifier",
#         environment_name="ml-tasks",
#     )


# if __name__ == "__main__":
#     main()
