import matplotlib.pyplot as plt
import seaborn as sns

def plot_scores(min_clusters,max_clusters,silhouette_scores,calinski_scores,best_n_clusters_silhouette,best_n_clusters_calinski):
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
    plt.savefig('Plots/clustering_scores.png')
    plt.show()

def plot_histplot_percentile(dataset, threshold):
    sns.histplot(dataset)
    plt.axvline(threshold, color='r', linestyle='--',
                label=f'85th percentile ({threshold:.2f})')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of distances to cluster centers on trainingset')
    plt.savefig(f'Plots/clusertering_distances {dataset}.png')
    plt.show()