import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchsummary import summary
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")

for param in model.parameters():
  param.requires_grad = True
print(model)
# summary(model, (1024,))

def tokenize_function(examples):
  return tokenizer(examples["highlights"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["article", "highlights", "id"])

tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))


train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=10)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=10)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
  for batch in train_dataloader:
    batch = {k: v for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)