from transformers import TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Trainer
from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from accelerate import Accelerator


import numpy as np
import evaluate
import nltk
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer
from ast import literal_eval


def train():
    batch_size = 4
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained("t5-small", add_prefix_space=True)
    model_checkpoint = "ml6team/keyphrase-generation-t5-small-inspec"

    training_args = Seq2SeqTrainingArguments(
        output_dir="inspec",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
    )

    # Dataset parameters
    dataset_document_column = "document"

    keyphrase_sep_token = ";"


    def preprocess_fuction(samples):
        processed_samples = {"input_ids": [], "attention_mask": [], "labels": []}
        for i, sample in enumerate(samples[dataset_document_column]):
            input_text = " ".join(sample)
            inputs = tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
            )
            keyphrases = samples["keywords"][i]

            target_text = f" {keyphrase_sep_token} ".join(keyphrases)

            with tokenizer.as_target_tokenizer():
                targets = tokenizer(
                    target_text, max_length=40, padding="max_length", truncation=True
                )
                targets["input_ids"] = [
                    (t if t != tokenizer.pad_token_id else -100)
                    for t in targets["input_ids"]
                ]
            for key in inputs.keys():
                processed_samples[key].append(inputs[key])
            processed_samples["labels"].append(targets["input_ids"])
        return processed_samples

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/midas/ldkp100_local/train.json",
        }
    )
    # Preprocess dataset
    tokenized_dataset = dataset.map(preprocess_fuction, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42)

    train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=small_train_dataset,
    )

    trainer.train()

    trainer.save_model("model/ldkp100_finetuning")

if __name__ == "__main__":
    train()