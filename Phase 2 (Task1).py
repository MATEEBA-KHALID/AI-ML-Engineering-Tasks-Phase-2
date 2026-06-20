from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import gradio as gr
import os

# =========================
# Load Dataset
# =========================

print("Loading AG News Dataset...")
dataset = load_dataset("ag_news")

# =========================
# Load Tokenizer
# =========================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =========================
# Tokenization
# =========================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# =========================
# Load Model
# =========================

model_path = "news_classifier_model"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = BertForSequenceClassification.from_pretrained(model_path)
else:
    print("Loading pretrained BERT...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )

# =========================
# Metrics
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }

# =========================
# Training
# =========================

if not os.path.exists(model_path):

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    print("Training started...")
    trainer.train()

    print("Evaluating model...")
    results = trainer.evaluate()

    print("\nResults:")
    print(results)

    print("Saving model...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# =========================
# Prediction Pipeline
# =========================

classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path
)

label_map = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech"
}

def predict_news(text):
    result = classifier(text)[0]

    label = label_map[result["label"]]
    confidence = round(result["score"] * 100, 2)

    return f"Category: {label}\nConfidence: {confidence}%"

# =========================
# Gradio Interface
# =========================

demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter a news headline..."
    ),
    outputs="text",
    title="News Topic Classifier Using BERT",
    description="Classifies news headlines into World, Sports, Business, or Sci/Tech."
)

if __name__ == "__main__":
    demo.launch()