import os, sys

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_path)

import argparse
import random
import math
import glob

import numpy as np
import tensorflow as tf
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.datasets import DatasetFactory
from experiments.utils import get_file_format
from experiments.preprocessing import load_image_from_file
from embedding_network_extractor import main as extract

tf.enable_eager_execution()

DATASETS = {
    "vggface2": "train", 
    "synth": None
    }

def get_image_paths_per_dataset(datasets_dir, cache_directory, n_per_dataset):
    image_paths_per_dataset = {}
    print("Getting Datasets...")
    for d in DATASETS:
        # Get the full dataset.
        print("Dataset {}:".format(d))
        dataset = DatasetFactory(datasets_dir, cache_dir=cache_directory).getDataset(d).get_dataset()
        
        # Use a subset if specified.
        if DATASETS[d]:
            dataset = dataset[DATASETS[d]]
            print("\tUsing {} subset: {}".format(d, DATASETS[d]))

        # Sample
        print("\tSampling {} images from dataset.".format(n_per_dataset))
        random.seed(42)
        image_paths_per_dataset[d] = random.choices(dataset["image_paths"], k=n_per_dataset)
        
    return image_paths_per_dataset


def create_dataset(image_paths):
    file_format = get_file_format(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image_from_file(file_format, 244, 244))
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(1)
    return dataset


def combine_datasets(tf_datasets):
    combined_dataset = None
    for d, dataset in tf_datasets.items():
        if not combined_dataset:
            combined_dataset = dataset.map(lambda img: (img, d))
        else:
            combined_dataset= combined_dataset.concatenate(dataset.map(lambda img: (img, d)))

    
def create_combined_dataset(imagepaths_per_dataset):
    tf_datasets = {d : create_dataset(imgs) for d, imgs in imagepaths_per_dataset.items()}
    return combine_datasets(tf_datasets)
    

def main(in_path, out_path, is_checkpoint, datasets_directory, force, n_per_dataset, cache_directory):
    print("Embedding Visualiser")
    in_path = os.path.abspath(os.path.expanduser(in_path))

    if not out_path:
        out_path = os.path.join("~", "embedding_visualisatiions", str(datetime.now()))
    out_path = os.path.abspath(os.path.expanduser(out_path))

    datasets_directory = os.path.abspath(os.path.expanduser(datasets_directory))
    cache_directory = os.path.abspath(os.path.expanduser(cache_directory))

    print("Args:")
    print("\tInput path:         ", in_path)
    print("\tOutput filepath:    ", out_path)
    print("\tDatasets directory: ", datasets_directory)
    print("\tCache directory:    ", cache_directory)

    if os.path.isdir(in_path):
        model_paths = sorted(glob.glob(in_path + "/*.hdf5"))
    else:
        model_paths = sorted([in_path])

    print("Models Found:")
    for model_path in model_paths:
        print("\t" + os.path.basename(model_path))
    
    os.makedirs(out_path, exist_ok=force)

    # Get datasets.
    imagepaths_per_dataset = get_image_paths_per_dataset(datasets_directory, cache_directory, n_per_dataset)

    # Get the tf_datasets.
    tf_datasets = {d : create_dataset(imgs) for d, imgs in imagepaths_per_dataset.items()}

    # Store all the charts.
    charts = {}

    # Iterate over the models.
    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print("Processing {}".format(model_name))
        # Get the model
        if is_checkpoint:
            embedding_model_path = os.path.join(out_path, "{}_embedding_model.hd5".format(model_name))
            extract(model_path, embedding_model_path)
            model_path = embedding_model_path
        model = tf.keras.models.load_model(model_path)

        # Get embeddings.
        steps = math.ceil(n_per_dataset / 32)
        embeddings_per_dataset = {d: model.predict(dataset, steps=steps) for d, dataset in tf_datasets.items()}

        # Combine the embeddings into a single numpy array.
        dataset_names = list(sorted(embeddings_per_dataset))
        embeddings = np.concatenate([embeddings_per_dataset[d] for d in dataset_names])
        labels = np.concatenate([np.repeat(d, len(embeddings_per_dataset[d])) for d in dataset_names])

        # Run PCA to reduce embeddings down to 50-d.
        embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings)

        # Run T-SNE to reduce down to 2-d.
        #embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

        # Create the pandas dataset.
        df = pd.DataFrame(data=np.c_[embeddings, labels], columns=["x", "y", "label"], )
        df["x"] = df["x"].astype('float32')
        df["y"] = df["y"].astype('float32')
        df["label"] = df["label"].astype('object')

        # Plot
        charts[model_name] = sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, legend=True, legend_out=True)
    
    # Get the max axis limit across all charts.
    max_y = min_y = max_x = min_x = None
    for c in charts.values():
        y1, y2 = c.ax.get_ylim()
        x1, x2 = c.ax.get_xlim()

        min_y = y1 if min_y is None or y1 < min_y else min_y
        max_y = y2 if max_y is None or y2 > max_y else max_y
        min_x = x1 if min_x is None or x1 < min_x else min_x
        max_x = x2 if max_x is None or x2 > max_x else max_x

    # Save all the charts.
    for c in charts:
        charts[c].ax.set(ylim=(min_y, max_y), xlim=(min_x, max_x))
        charts[c].savefig(os.path.join(out_path, "chart_{}.png".format(c)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tries to visualise ')
    parser.add_argument("input", help="Path to the checkpoint file or direcoty.")
    parser.add_argument("--output-path", "-o")
    parser.add_argument("--checkpoint-file", "-c", action="store_true")
    parser.add_argument("--dataset-directory", "-d", default="~/datasets")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--n-per-dataset", "-n", default=500)
    parser.add_argument("--cache-directory", default="~/cache")
    args = parser.parse_args()

    main(args.input, args.output_path, args.checkpoint_file, args.dataset_directory, args.force, args.n_per_dataset, args.cache_directory)