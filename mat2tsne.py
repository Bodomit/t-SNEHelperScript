import os
import argparse
import random
import math
import glob

from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main(in_path, out_path, perplexity):
    print("Embedding Visualiser")
    in_path = os.path.abspath(os.path.expanduser(in_path))

    if not out_path:
        out_path = os.path.join("~", "embedding_visualisations", str(datetime.now()))
    out_path = os.path.abspath(os.path.expanduser(out_path))

    print("Args:")
    print("\tInput path:         ", in_path)
    print("\tOutput filepath:    ", out_path)
    print("\tPerplexity:         ", perplexity)

    # Create the output directory.
    os.makedirs(out_path)

    # Load the embeddings from mat file.
    embeddings = sio.loadmat(in_path)

    # Run T-SNE to reduce down to 2-d.
    embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

    # Create the pandas dataset.
    df = pd.DataFrame(data=embeddings, columns=["x", "y"])
    df["x"] = df["x"].astype('float32')
    df["y"] = df["y"].astype('float32')

    # Plot
    chart = sns.lmplot(data=df, x='x', y='y')
    chart.savefig(os.path.join(out_path, "tsne_p{}.png".format(perplexity)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tries to visualise ')
    parser.add_argument("input-path", help="Path to the checkpoint file or direcoty.")
    parser.add_argument("--output-path", "-o")
    parser.add_argument("--perplexity", "-p", default=30)
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.perplexity)