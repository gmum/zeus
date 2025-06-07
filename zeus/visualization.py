from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_tsne(output, X, y_true, plots_path, seed):
    X_np = X.numpy()
    y_true_np = y_true.numpy()

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=7, random_state=seed)
    #print("X_np: ", X_np.shape)
    #tsne = PCA(n_components=2)
    out_dim = tsne.fit_transform(X_np)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=7, random_state=seed)
    #print("output: ", output.shape)
    #tsne = PCA(n_components=2)
    out_dim2 = tsne.fit_transform(output)

    cmap = plt.cm.get_cmap('tab10', 10)
    colors_y_true = cmap(y_true_np)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(out_dim[:, 0], out_dim[:, 1], c=colors_y_true, s=10, alpha=0.7)
    #axes[0].set_title("Input space")
    axes[0].axis('off')

    axes[1].scatter(out_dim2[:, 0], out_dim2[:, 1], c=colors_y_true, s=10, alpha=0.7)
    #axes[1].set_title("ZEUS representation")
    axes[1].axis('off')

    # Get the positions in figure coordinates
    posA = axes[0].get_position()
    posB = axes[1].get_position()

    start = (5*posA.x1/6 + 0.02, 5 * (posA.y0 + posA.y1) / 6 - 0.03)
    end = (posB.x0 + 1*posA.x1/6 - 0.02, 5 * (posB.y0 + posB.y1) / 6 - 0.03)

    # Add arrow
    arrow = FancyArrowPatch(start, end,
                            transform=fig.transFigure,
                            connectionstyle="arc3,rad=-0.3",
                            arrowstyle='->',
                            mutation_scale=25,
                            color='black',
                            linewidth=4)

    fig.patches.append(arrow)

    # Add label above the arrow
    mid_x = (start[0] + end[0]) / 2
    mid_y = 5 * (posB.y0 + posB.y1) / 6 + 0.05
    fig.text(mid_x, mid_y, "ZEUS", ha='center', va='bottom', fontsize=24)

    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(plots_path, dpi=300)
    plt.close()
