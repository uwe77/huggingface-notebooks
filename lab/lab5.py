from datasets import load_dataset, DatasetDict
import os
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import shutil


def upload_dataset(dataset_path="data/wine-dataset.csv", 
                   dataset_type="csv", 
                   dataset_repo="uwwee/sis_lab5"):
    
    if os.path.isfile(dataset_path):
        print("File exists")
    else:
        print("File does not exist")
        return
    
    dataset = load_dataset(dataset_type, data_files=dataset_path, delimiter=",")
    print(dataset)
    dataset = dataset.rename_column(
        original_column_name="Unnamed: 0", new_column_name="wine_id"
    )
    print(dataset)
    dataset = dataset.remove_columns(
	    column_names=["province", "region_1", "region_2", "taster_name", "taster_twitter_handle", "title", "variety", "winery"]
    )
    print(dataset)
    def to_lowercase(example):
        example['description'] = example['description'].lower()
        example['country'] = example['country'].lower()
        return example
    dataset = dataset.map(to_lowercase)
    split_dataset = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    print(train_dataset)
    print(test_dataset)	
    wine_sample = train_dataset.shuffle(seed=42).select(range(10))
    # Peek at the first few examples
    print(wine_sample["wine_id"])
    print(wine_sample["country"])
    italian_wines = train_dataset.filter(lambda example: example['country'] == 'italy')
    print("\nFiltered Italian wines sample:")
    print(italian_wines[:3])
    # Example: Select wines with more than 90 points
    highly_rated_wines = train_dataset.filter(lambda example: example['points'] > 90)
    highly_rated_wines = highly_rated_wines.sort("points", reverse=True)
    print("\nHighly rated wines sample:")
    print(highly_rated_wines[:3])
    train_dataset.set_format("pandas")
    print(train_dataset[:3])
    train_df = train_dataset[:]
    frequencies = (
        train_df["points"]
        .value_counts()
        .to_frame()
        .reset_index()
        .rename(columns={"count": "frequency"})
    )
    frequencies.head()
    train_dataset.reset_format()
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    dataset_dict.push_to_hub(dataset_repo)

def download_dataset(dataset_path="data/wine-dataset.csv", 
                     dataset_type="csv", 
                     dataset_repo="uwwee/sis_lab5"):
    
    # dataset = load_dataset(dataset_type, dataset_repo)
    dataset = load_dataset(dataset_repo)
    # dataset.save_to_disk(dataset_path)
    print(dataset)

if __name__ == "__main__":
    upload_dataset()
    download_dataset()