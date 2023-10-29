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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def text():
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




def image():
    folder_path_image = "temp/text/image"
    folderExist = os.path.exists(folder_path_image)
    if not folderExist:
        os.makedirs(folder_path_image)

text()

# # Initialize the tokenizer and model


# print(f"Tokenizer and model saved to {output_dir}")




# shutil.rmtree(folder_path, ignore_errors=True)


# project_id = 'mlops-398205'
# bucket_name = 'registry-tamlops'
# source_file_path = '/root/tamlops/service/pipeline/random_forest_model.pkl'
# destination_blob_name = 'Text-Model'


    
# client = storage.Client(project=project_id)

# bucket = client.get_bucket(bucket_name)

# blob = bucket.blob(destination_blob_name)

# blob.upload_from_filename(source_file_path)

# print(f"File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}.")


