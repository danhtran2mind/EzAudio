import datasets
import pandas as pd
from huggingface_hub import snapshot_download
import os
import shutil
import tarfile
import argparse
import logging
import multiprocessing as mp
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_clean_dataset(dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the MusicBench dataset, removing duplicates by keeping the row
    with the longest main_caption per location.

    Args:
        dataset_id (str): Hugging Face dataset ID.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned train and validation DataFrames.
    """
    try:
        dataset = datasets.load_dataset(dataset_id)
        if "train" not in dataset or "test" not in dataset:
            raise ValueError("Dataset does not contain 'train' or 'test' splits.")
        
        train_df = pd.DataFrame(dataset["train"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        val_df = pd.DataFrame(dataset["test"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        
        # Split validation set into val and test (50-50 split)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = len(val_df) // 2
        val_df, test_df = val_df.iloc[:split_idx], val_df.iloc[split_idx:]
        
        logger.info(f"Train set size: {len(train_df)}, Val set size: {len(val_df)}, Test set size: {len(test_df)}")
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"Failed to load or clean dataset {dataset_id}: {e}")
        raise

def download_and_extract_dataset(dataset_id: str, raw_data_dir: str, music_bench_dir: str) -> None:
    """
    Download and extract the MusicBench dataset from Hugging Face.

    Args:
        dataset_id (str): Hugging Face dataset ID.
        raw_data_dir (str): Directory to store raw downloaded data.
        music_bench_dir (str): Directory to extract the dataset.

    Raises:
        FileNotFoundError: If the tar file is not found.
        Exception: For other download or extraction errors.
    """
    try:
        os.makedirs(music_bench_dir, exist_ok=True)
        snapshot_download(repo_id=dataset_id, local_dir=raw_data_dir, repo_type="dataset")
        
        tar_path = os.path.join(raw_data_dir, "MusicBench.tar.gz")
        
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=music_bench_dir)
        logger.info(f"Dataset extracted to {music_bench_dir}")
    except Exception as e:
        logger.error(f"Failed to download or extract dataset: {e}")
        raise

def move_file(args: Tuple[str, str, int]) -> Tuple[int, bool]:
    """
    Move a single file and handle errors.

    Args:
        args (Tuple[str, str, int]): Source path, destination path, and index.

    Returns:
        Tuple[int, bool]: Index and success status.
    """
    src_path, dst_path, index = args
    try:
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)
            return index, True
        else:
            logger.warning(f"File not found: {src_path}")
            return index, False
    except Exception as e:
        logger.error(f"Error moving file {src_path} to {dst_path}: {e}")
        return index, False

def move_and_cleanup_files(
    raw_data_dir: str,
    music_bench_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_processes: int,
    dataset_name: str,
    data_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Move files from datashare to train/val/test folders and clean up.

    Args:
        raw_data_dir (str): Directory containing raw downloaded data.
        music_bench_dir (str): Directory containing extracted dataset.
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        num_processes (int): Number of processes for parallel file moving.
        dataset_name (str): Dataset directory name.
        data_dir (str): Base directory for dataset storage.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated train, val, and test DataFrames.
    """
    try:
        # Create train, val, and test directories
        train_dir = os.path.join(data_dir, dataset_name, "train")
        val_dir = os.path.join(data_dir, dataset_name, "val")
        test_dir = os.path.join(data_dir, dataset_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        datashare_dir = os.path.join(music_bench_dir, "datashare")
        if not os.path.exists(datashare_dir):
            raise FileNotFoundError(f"Datashare directory not found: {datashare_dir}")

        # Prepare arguments for parallel file moving
        train_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(train_dir, os.path.basename(row["location"])),
             index)
            for index, row in train_df.iterrows()
        ]
        val_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(val_dir, os.path.basename(row["location"])),
             index)
            for index, row in val_df.iterrows()
        ]
        test_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(test_dir, os.path.basename(row["location"])),
             index)
            for index, row in test_df.iterrows()
        ]

        # Use multiprocessing pool for file moving
        with mp.Pool(processes=num_processes) as pool:
            train_results = pool.map(move_file, train_tasks)
            for index, success in train_results:
                if not success:
                    train_df = train_df.drop(index, errors="ignore")
            
            val_results = pool.map(move_file, val_tasks)
            for index, success in val_results:
                if not success:
                    val_df = val_df.drop(index, errors="ignore")
            
            test_results = pool.map(move_file, test_tasks)
            for index, success in test_results:
                if not success:
                    test_df = test_df.drop(index, errors="ignore")

        # Remove files not in train_df
        train_filenames = set(os.path.basename(row["location"]) for _, row in train_df.iterrows())
        for filename in os.listdir(train_dir):
            if filename not in train_filenames:
                file_path = os.path.join(train_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from train folder: {file_path}")

        # Remove files not in val_df
        val_filenames = set(os.path.basename(row["location"]) for _, row in val_df.iterrows())
        for filename in os.listdir(val_dir):
            if filename not in val_filenames:
                file_path = os.path.join(val_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from val folder: {file_path}")

        # Remove files not in test_df
        test_filenames = set(os.path.basename(row["location"]) for _, row in test_df.iterrows())
        for filename in os.listdir(test_dir):
            if filename not in test_filenames:
                file_path = os.path.join(test_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from test folder: {file_path}")

        # Clean up directories
        for dir_path in [music_bench_dir, raw_data_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.info(f"Cleaned up directory: {dir_path}")
        
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"Error in move_and_cleanup_files: {e}")
        raise

def write_csv_files(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    data_dir: str
) -> None:
    """
    Write train, val, and test data to CSV files with the specified structure.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        dataset_name (str): Dataset directory name.
        data_dir (str): Base directory for dataset storage.

    Raises:
        Exception: If writing CSV files fails.
    """
    try:
        csv_dir = os.path.join(data_dir, dataset_name)
        os.makedirs(csv_dir, exist_ok=True)

        # Prepare CSV data
        train_csv = pd.DataFrame({
            "audio_path": [f"{dataset_name}/train/{os.path.basename(row['location'])}" for _, row in train_df.iterrows()],
            "caption": train_df["main_caption"],
            "split": "train"
        })
        val_csv = pd.DataFrame({
            "audio_path": [f"{dataset_name}/val/{os.path.basename(row['location'])}" for _, row in val_df.iterrows()],
            "caption": val_df["main_caption"],
            "split": "val"
        })
        test_csv = pd.DataFrame({
            "audio_path": [f"{dataset_name}/test/{os.path.basename(row['location'])}" for _, row in test_df.iterrows()],
            "caption": test_df["main_caption"],
            "split": "test"
        })

        # Write CSV files
        train_csv.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
        val_csv.to_csv(os.path.join(csv_dir, "val.csv"), index=False)
        test_csv.to_csv(os.path.join(csv_dir, "test.csv"), index=False)
        logger.info(f"CSV files written to {csv_dir}")
    except Exception as e:
        logger.error(f"Error writing CSV files: {e}")
        raise

def main(arg_process: int, dataset_id: str, data_dir: str) -> None:
    """
    Main function to execute the dataset processing pipeline and create CSV files.

    Args:
        arg_process (int): Number of processes for parallel operations.
        dataset_id (str): Hugging Face dataset ID.
        data_dir (str): Base directory for dataset storage.

    Raises:
        ValueError: If arguments are invalid.
        Exception: For other processing errors.
    """
    try:
        # Validate arguments
        if arg_process < 1:
            raise ValueError("Number of processes must be at least 1.")
        if not dataset_id:
            raise ValueError("Dataset ID cannot be empty.")
        if not data_dir:
            raise ValueError("Data directory cannot be empty.")

        # Ensure data_dir exists
        os.makedirs(data_dir, exist_ok=True)

        # Limit the number of processes
        num_processes = min(arg_process, os.cpu_count() or 1)
        logger.info(f"Using {num_processes} processes for parallel file operations.")

        # Process dataset_id to create directory name
        dataset_name = dataset_id.replace("/", "-")
        raw_data_dir = os.path.join(data_dir, f"datasets--{dataset_name}")
        music_bench_dir = os.path.join(data_dir, dataset_name, "music_bench")

        # Execute pipeline
        train_df, val_df, test_df = load_and_clean_dataset(dataset_id)
        download_and_extract_dataset(dataset_id, raw_data_dir, music_bench_dir)
        train_df, val_df, test_df = move_and_cleanup_files(raw_data_dir, music_bench_dir, 
                                                          train_df, val_df, test_df, 
                                                          num_processes, dataset_name, 
                                                          data_dir)
        write_csv_files(train_df, val_df, test_df, dataset_name, data_dir)

        logger.info("Dataset processing and CSV creation completed successfully.")
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MusicBench dataset and create CSV files.")
    parser.add_argument(
        "--arg_process",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of processes to use (capped at CPU count)."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="amaai-lab/MusicBench",
        help="Dataset ID for Hugging Face dataset (default: amaai-lab/MusicBench)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for dataset storage (default: data)"
    )
    args = parser.parse_args()
    main(args.arg_process, args.dataset_id, args.data_dir)
