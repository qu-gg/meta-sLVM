"""
@file plotting.py

Holds general plotting functions for reconstructions of the bouncing ball dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.manifold import TSNE
from scipy import interpolate
from scipy.spatial import ConvexHull


def show_images(images, preds, out_loc, num_out=None):
    """
    Constructs an image of multiple time-series reconstruction samples compared against its relevant ground truth
    Saves locally in the given out location
    :param images: ground truth images
    :param preds: predictions from a given model
    :out_loc: where to save the generated image
    :param num_out: how many images to stack. If None, stack all
    """
    assert len(images.shape) == 4       # Assert both matrices are [Batch, Timesteps, H, W]
    assert len(preds.shape) == 4
    assert type(num_out) is int or type(num_out) is None

    # Make sure objects are in numpy format
    if not isinstance(images, np.ndarray):
        images = images.cpu().numpy()
        preds = preds.cpu().numpy()

    # Splice to the given num_out
    if num_out is not None:
        images = images[:num_out]
        preds = preds[:num_out]

    # Iterate through each sample, stacking into one image
    out_image = None
    for idx, (gt, pred) in enumerate(zip(images, preds)):
        # Pad between individual timesteps
        gt = np.pad(gt, pad_width=(
            (0, 0), (5, 5), (0, 1)
        ), constant_values=1)

        gt = np.hstack([i for i in gt])

        # Pad between individual timesteps
        pred = np.pad(pred, pad_width=(
            (0, 0), (0, 10), (0, 1)
        ), constant_values=1)

        # Stack timesteps into one image
        pred = np.hstack([i for i in pred])

        # Stack gt/pred into one image
        final = np.vstack((gt, pred))

        # Stack into out_image
        if out_image is None:
            out_image = final
        else:
            out_image = np.vstack((out_image, final))

    # Save to out location
    plt.imsave(out_loc, out_image, cmap='gray')


def get_embedding_hist(output_path, args, title, stacked_set, value):
    """
    For all dynamics and their associated embeddings, get an overlapping histogram
    :param output_path: full path for saving
    :param title: whether the embeddings are the distribuional parameters or samples
    """
    # Extract sets
    embeddings, labels = stacked_set[value], stacked_set["labels"]

    # Histogram them
    for i in np.unique(labels):
        subset = np.reshape(embeddings[np.where(labels == i)[0], :], [-1])
        plt.hist(subset, bins=100, alpha=0.5, color=args.colors[int(i)], label=f"{i}")

    plt.legend()
    plt.title(f"Histogram of Generated {title}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def get_embedding_tsneD(output_path, args, train_set, test_set):
    # Extract sets
    train_embeddings, train_labels = train_set["embeddings"], train_set["labels"]
    test_embeddings, test_labels = test_set["embeddings"], test_set["labels"]

    # Ensure embeddings are flattened
    if len(train_embeddings.shape) > 2:
        train_embeddings = train_embeddings.reshape([train_embeddings.shape[0] * train_embeddings.shape[1], -1])
        test_embeddings = test_embeddings.reshape([test_embeddings.shape[0] * test_embeddings.shape[1], -1])

    # Define TSNE object
    if train_embeddings.shape[-1] > 2:
        tsne = TSNE(n_components=2, perplexity=30, metric="cosine", n_iter=3000, random_state=3)
        tsne_embedding = tsne.fit_transform(np.vstack((train_embeddings, test_embeddings)))
        train_embeddings = tsne_embedding[:train_embeddings.shape[0]]
        test_embeddings = tsne_embedding[train_embeddings.shape[0]:]

    # Plot codes in TSNE
    plt.figure()

    # Plot codes in TSNE
    markers = ['o', 'v', '^', '<', '>', 's', '8', 'p', 'o', 'v', '^', '<', '>', 's', '8', 'p']
    for c_idx, i in enumerate(np.unique(test_labels)):
        for c, tsne_embedding, labels in zip(
                ['k', 'off'],
                [train_embeddings, test_embeddings],
                [train_labels, test_labels]
        ):
            color = args.colors[int(i)] if c == 'off' else 'k'

            # Get subset
            subset = tsne_embedding[np.where(labels == i)[0], :]
            if subset.shape[0] == 0:
                continue

            # # Get convex hull
            # hull = ConvexHull(subset)
            # x_hull = np.append(subset[hull.vertices, 0], subset[hull.vertices, 0][0])
            # y_hull = np.append(subset[hull.vertices, 1], subset[hull.vertices, 1][0])
            #
            # # Interpolate
            # dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
            # dist_along = np.concatenate(([0], dist.cumsum()))
            # spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
            # interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            # interp_x, interp_y = interpolate.splev(interp_d, spline)

            # Plot points in cluster
            plt.scatter(subset[:, 0], subset[:, 1], alpha=0.5, c=color, marker=markers[i], label=f"{i}")

            # Plot boundaries
            # plt.fill(interp_x, interp_y, '--', alpha=0.2, c=color)

    # Save it without topology
    plt.title("t-SNE Plot of Generated Codes")
    plt.legend(loc='upper right')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def get_embedding_tsne(output_path, args, train_set, test_set):
    # Extract sets
    train_embeddings, train_labels = train_set["embeddings"], train_set["labels"]
    test_embeddings, test_labels = test_set["embeddings"], test_set["labels"]

    # Ensure embeddings are flattened
    if len(train_embeddings.shape) > 2:
        train_embeddings = train_embeddings.reshape([train_embeddings.shape[0] * train_embeddings.shape[1], -1])
        test_embeddings = test_embeddings.reshape([test_embeddings.shape[0] * test_embeddings.shape[1], -1])

    # Define TSNE object
    if train_embeddings.shape[-1] > 2:
        tsne = TSNE(n_components=2, perplexity=30, metric="cosine", n_iter=3000, random_state=3)
        tsne_embedding = tsne.fit_transform(np.vstack((train_embeddings, test_embeddings)))
        train_embeddings = tsne_embedding[:train_embeddings.shape[0]]
        test_embeddings = tsne_embedding[train_embeddings.shape[0]:]

    # Create a figure with 1 row and 4 columns (1x4 grid)
    fig = plt.figure(figsize=(10, 8), dpi=300)

    # Plot codes in TSNE
    markers = ['o', 'v', '^', '<', '>', 's', '8', 'p', 'o', 'v', '^', '<', '>', 's', '8', 'p']
    for c_idx, i in enumerate(np.unique(test_labels)):
        for c, tsne_embedding, labels in zip(
                ['k', 'off'],
                [train_embeddings, test_embeddings],
                [train_labels, test_labels]
        ):
            color = args.colors[int(i)] if c == 'off' else 'k'

            # Get subset
            subset = tsne_embedding[np.where(labels == i)[0], :]
            if subset.shape[0] == 0:
                continue

            # Plot points in cluster
            plt.scatter(subset[:, 0], subset[:, 1], alpha=0.5, c=color, marker=markers[i], s=25)

    import matplotlib.lines as mlines
    bb_patch1 = mlines.Line2D([], [], marker=markers[0], linestyle='None', markersize=10, color=args.colors[0], label='Gravity N')
    bb_patch2 = mlines.Line2D([], [], marker=markers[1], linestyle='None', markersize=10, color=args.colors[1], label='Gravity SW')
    bb_patch3 = mlines.Line2D([], [], marker=markers[2], linestyle='None', markersize=10, color=args.colors[2], label='Gravity SE')
    pd_patch1 = mlines.Line2D([], [], marker=markers[3], linestyle='None', markersize=10, color=args.colors[3], label='Pendulum 2G')
    pd_patch2 = mlines.Line2D([], [], marker=markers[4], linestyle='None', markersize=10, color=args.colors[4], label='Pendulum 3G')
    pd_patch3 = mlines.Line2D([], [], marker=markers[5], linestyle='None', markersize=10, color=args.colors[5], label='Pendulum 4G')
    dp_patch1 = mlines.Line2D([], [], marker=markers[6], linestyle='None', markersize=10, color=args.colors[6], label='Mass Spring 1K')
    dp_patch2 = mlines.Line2D([], [], marker=markers[7], linestyle='None', markersize=10, color=args.colors[7], label='Mass Spring 2K')
    dp_patch3 = mlines.Line2D([], [], marker=markers[8], linestyle='None', markersize=10, color=args.colors[8], label='Mass Spring 3K')
    tb_patch1 = mlines.Line2D([], [], marker=markers[9], linestyle='None', markersize=10, color=args.colors[9], label='Two Body 1G')
    tb_patch2 = mlines.Line2D([], [], marker=markers[10], linestyle='None', markersize=10, color=args.colors[10], label='Two Body 2G')
    tb_patch3 = mlines.Line2D([], [], marker=markers[11], linestyle='None', markersize=10, color=args.colors[11], label='Two Body 3G')
    handles = [bb_patch1, bb_patch2, bb_patch3, pd_patch1, pd_patch2, pd_patch3, dp_patch1, dp_patch2, dp_patch3, tb_patch1, tb_patch2, tb_patch3]
    labels = ['Gravity N', 'Gravity SW', 'Gravity SE', 'Pendulum 2G', 'Pendulum 3G', 'Pendulum 4G', 'Mass Spring 1K', 'Mass Spring 2K', 'Mass Spring 3K', 'Two Body 1G', 'Two Body 2G', 'Two Body 3G']
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4, fontsize=14, frameon=True)

    # Customize legend to block out lines behind it
    legend.get_frame().set_facecolor('white')  # Set solid white background
    legend.get_frame().set_edgecolor('black')  # Add black border around the legend
    legend.get_frame().set_alpha(1)  # Ensure no transparency

    # Adjust layout to prevent overlap and make room for the shared legend
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Increased bottom space for the legend

    # Save it without topology
    plt.title("t-SNE Plot of Generated Codes")
    plt.legend(loc='upper right')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

