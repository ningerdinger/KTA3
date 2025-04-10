import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_with_incremental_filename(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"

def plot_scores(min_clusters, max_clusters, silhouette_scores, calinski_scores, best_n_clusters_silhouette, best_n_clusters_calinski):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', label='Silhouette Score')
    plt.axvline(x=best_n_clusters_silhouette, color='r', linestyle='--', label=f'Best Clustering ({best_n_clusters_silhouette})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.legend()
    plt.title('Silhouette score vs number of clusters')

    plt.subplot(1, 2, 2)
    plt.plot(range(min_clusters, max_clusters + 1), calinski_scores, marker='o', label='Calinski-Harabasz Score')
    plt.axvline(x=best_n_clusters_calinski, color='r', linestyle='--', label=f'Best Clustering ({best_n_clusters_calinski})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz score')
    plt.legend()
    plt.title('Calinski-Harabasz score vs number of clusters')

    plt.tight_layout()
    save_path = save_with_incremental_filename('Plots/clustering_scores.png')
    plt.savefig(save_path)
    plt.show()

def plot_histplot_percentile(dataset, threshold):
    sns.histplot(dataset)
    plt.axvline(threshold, color='r', linestyle='--',
                label=f'85th percentile of TrainingSet({threshold:.2f})')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of distances to cluster centers')
    save_path = save_with_incremental_filename('Plots/clusertering_distances.png')
    plt.savefig(save_path)
    plt.show()

def plot_actor_presence(count_df, value_col, cluster_choice, title):
    sns.histplot(data=count_df, x=value_col, bins=cluster_choice + 1, discrete=True)
    plt.title(f'Presence Count of Actors in {title}')
    save_path = save_with_incremental_filename(f'Plots/Presence in {title} dataset.png')
    plt.savefig(save_path)
    # Adjust x-axis labels to include 'outlier' followed by cluster numbers
    plt.xticks(ticks=range(-1,cluster_choice), labels=['outlier'] + list(range(cluster_choice)))
    plt.show()

def plot_cluster_presence(cluster_dataframe):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_dataframe, cmap="YlGnBu", cbar=False, linewidths=0.5)
    plt.title("Cluster Presence Across Frames")
    plt.xlabel("Frame")
    plt.ylabel("Cluster")
    row_labels = cluster_dataframe.index.tolist()
    row_labels = ['outlier' if label == -1 else label for label in row_labels]
    plt.yticks(ticks=range(len(row_labels)), labels=row_labels)
    save_path = save_with_incremental_filename('Plots/Heatmap.png')
    plt.savefig(save_path)
    plt.show()