import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from google.cloud import storage
from transformers import T5Tokenizer, T5ForConditionalGeneration
import shutil
from datasets import load_dataset, concatenate_datasets
from random import randrange  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from huggingface_hub import HfFolder



def train_image(datas):
    print("")

def train_text(datas):
    def preprocess_function(sample,padding="max_length"):
        inputs = ["summarize: " + item for item in sample["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    
    # Initial Setup
    folder_path_text = "temp/text"
    folderExist = os.path.exists(folder_path_text)
    if not folderExist:
        os.makedirs(folder_path_text)

    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    # # Specify the directory where you want to save the tokenizer and model
    output_dir = "temp/text"

    # Load dataset from the hub
    dataset_id = "samsum"
    dataset = load_dataset(dataset_id)
    num_train_samples = 50
    dataset['train'] = dataset['train'].select(range(num_train_samples))

    num_test_samples = 10
    dataset['test'] = dataset['test'].select(range(num_test_samples))
    print(dataset['train'])


    model_id="google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # # The maximum total input sequence length after tokenization. 
    # # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # # The maximum total sequence length for target text after tokenization. 
    # # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")


    # tokenizer.save_pretrained(output_dir)
    # model.save_pretrained(output_dir)

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=1
    )
    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="/root/tamlops/service/pipeline/example/tmp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        # learning_rate=5e-5,
        num_train_epochs=1,
        # logging & evaluation strategies
        # logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        # hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
    )
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        # compute_metrics=compute_metrics,
    )
    # Start training
    trainer.train()
    print("[!] Flan T5 Has been Trained")
    



train_text()