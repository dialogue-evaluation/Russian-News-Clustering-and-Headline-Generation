# Script for MLM BERT pretraining, used to build https://huggingface.co/IlyaGusev/news_tg_rubert
#
# Data:
# wget https://www.dropbox.com/s/bb7q2sh53fgyc21/all_texts.txt.tar.gz
# head -n 550000 all_texts.txt > train.txt
# tail -n 45064 all_texts.txt > val.txt
#
# Data includes:
# https://data-static.usercontent.dev/DataClusteringSample0107.tar.gz
# https://data-static.usercontent.dev/DataClusteringSample0817.tar.gz
# https://data-static.usercontent.dev/DataClusteringSample1821.tar.gz
# https://data-static.usercontent.dev/DataClusteringSample2225.tar.gz
# https://data-static.usercontent.dev/DataClusteringDataset.tar.gz
# https://data-static.usercontent.dev/DataClusteringDataset1209.tar.gz
# https://data-static.usercontent.dev/DataClusteringDataset0131.tar.gz
# https://data-static.usercontent.dev/DataClusteringDataset0214.tar.gz
#
# To reproduce all_texts.txt, see "Fetching texts for pretraining" section of baselines ipynb.

import argparse
import os
from typing import Dict

import razdel
import torch
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer, BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

class LineByLineTextDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 line_sentences: int=5):
        if tokenizer is None:
            return
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.isspace():
                    continue
                sentences = [s.text for s in razdel.sentenize(line)] 
                start_index = 0
                while start_index < len(sentences):
                    end_index = start_index + line_sentences
                    line = " ".join(sentences[start_index:end_index])
                    lines.append(line)
                    start_index = end_index

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = []
        for input_ids, attention_mask in zip(batch_encoding["input_ids"], batch_encoding["attention_mask"]):
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

    def save(self, path):
        torch.save(self.examples, path)

    @classmethod
    def load(cls, path):
        obj = cls(None, None, None)
        obj.examples = torch.load(path)
        return obj


def pretrain_mlm(
    initial_model_name,
    train_path,
    train_save_path,
    val_path,
    val_save_path,
    block_size,
    out_dir,
    eval_steps,
    batch_size,
    grad_accum_steps,
    epochs,
    lr
):
    tokenizer = BertTokenizer.from_pretrained(
        initial_model_name,
        do_lower_case=False,
        do_basic_tokenize=False,
        strip_accents=False
    )
    if os.path.exists(train_save_path) and os.path.exists(val_save_path):
        train_dataset = torch.load(train_save_path)
        val_dataset = torch.load(val_save_path)
    else:
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=train_path,
            block_size=block_size
        )

        val_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=val_path,
            block_size=block_size
        )

        train_dataset.save(train_save_path)
        val_dataset.save(val_save_path)

    for r in train_dataset:
        print(r)
        break

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        save_total_limit=2,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps"
    )

    model = BertForMaskedLM.from_pretrained(initial_model_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-model-name", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--train-path", type=str, default="train.txt")
    parser.add_argument("--val-path", type=str, default="val.txt")
    parser.add_argument("--train-save-path", type=str, default="train.pt")
    parser.add_argument("--val-save-path", type=str, default="val.pt")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=8)   
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-05)
    args = parser.parse_args()
    pretrain_mlm(**vars(args))
