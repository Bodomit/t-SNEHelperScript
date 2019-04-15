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


def main(in_path, out_path, n_components, perplexity):
    print("Embedding Visualiser")
    in_path = os.path.abspath(os.path.expanduser(in_path))

    if not out_path:
        out_path = os.path.join("~", "embedding_visualisations", str(datetime.now()))
    out_path = os.path.abspath(os.path.expanduser(out_path))

    print("Args:")
    print("\tInput path:         ", in_path)
    print("\tOutput filepath:    ", out_path)
    print("\tN components:       ", n_components)
    print("\tPerplexity:         ", perplexity)

    # Create the output directory.
    os.makedirs(out_path)

    # Load the embeddings from mat file. Get the first array in the matfile.
    matfiledata = sio.loadmat(in_path)
    for k in matfiledata:
        if isinstance(matfiledata[k], np.ndarray):
            embeddings = matfiledata[k]

    # Run T-SNE to reduce down to n-components.
    reduced_embeddings = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, verbose=1).fit_transform(embeddings)

    # Save as numpy array and matfile.
    print("Saving numpy array...")
    np.save(os.path.join(out_path, "TSNE_N{}_p{}.npy".format(n_components, perplexity)), reduced_embeddings)
    print("Saving matfile...")
    sio.savemat(os.path.join(out_path, "TSNE_N{}_p{}.mat".format(n_components, perplexity)), mdict={"tsne": reduced_embeddings})

    # Create the pandas dataset and plot.
    print("Plotting...")
    df = pd.DataFrame(data=embeddings)
    columns = df.columns.values.tolist()
    chart = sns.lmplot(data=df, x=columns[0], y=columns[1])
    chart.savefig(os.path.join(out_path, "tsne_p{}.png".format(perplexity)))
    print("Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tries to visualise ')
    parser.add_argument("input", help="Path to the checkpoint file or direcoty.")
    parser.add_argument("--output-path", "-o")
    parser.add_argument("--n-components", "-n", default=2)
    parser.add_argument("--perplexity", "-p", default=30.0)
    args = parser.parse_args()

    main(args.input, args.output_path, args.n_components, args.perplexity)