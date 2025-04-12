import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_with_incremental_filename(base_path):
    """
    Generates an incremental filename if the specified file already exists.

    Parameters:
        base_path (str): The desired file path including the name and extension.

    Returns:
        str: A unique file path by appending a counter to the base name if a file with the same name exists.
    """
    if not os.path.exists(base_path):  # Check if the file does not exist.
        return base_path  # Return the original file path if it's unique.
    base, ext = os.path.splitext(base_path)  # Split the file name and extension.
    counter = 1  # Initialize the counter.
    while os.path.exists(f"{base}_{counter}{ext}"):  # Increment the counter until a unique file name is found.
        counter += 1
    return f"{base}_{counter}{ext}"  # Return the unique file path.


def plot_scores(min_clusters, max_clusters, silhouette_scores, calinski_scores, best_n_clusters_silhouette, best_n_clusters_calinski):
    """
    Plots and saves the silhouette and calinski-harabasz scores for different cluster counts.

    Parameters:
        min_clusters (int): Minimum number of clusters tested.
        max_clusters (int): Maximum number of clusters tested.
        silhouette_scores (list): Silhouette scores for each cluster count.
        calinski_scores (list): Calinski-Harabasz scores for each cluster count.
        best_n_clusters_silhouette (int): Cluster count with the best silhouette score.
        best_n_clusters_calinski (int): Cluster count with the best calinski-harabasz score.

    Returns:
        None
    """
    plt.figure(figsize=(15, 5))  # Create a figure with two subplots.
    plt.subplot(1, 2, 1)  # First subplot for silhouette scores.
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', label='Silhouette Score')  # Plot silhouette scores.
    plt.axvline(x=best_n_clusters_silhouette, color='r', linestyle='--', label=f'Best Clustering ({best_n_clusters_silhouette})')  # Highlight best clustering.
    plt.xlabel('Number of clusters')  # Label for x-axis.
    plt.ylabel('Silhouette score')  # Label for y-axis.
    plt.legend()
    plt.title('Silhouette score vs number of clusters')

    plt.subplot(1, 2, 2)  # Second subplot for calinski-harabasz scores.
    plt.plot(range(min_clusters, max_clusters + 1), calinski_scores, marker='o', label='Calinski-Harabasz Score')  # Plot calinski-harabasz scores.
    plt.axvline(x=best_n_clusters_calinski, color='r', linestyle='--', label=f'Best Clustering ({best_n_clusters_calinski})')  # Highlight best clustering.
    plt.xlabel('Number of clusters')  # Label for x-axis.
    plt.ylabel('Calinski-Harabasz score')  # Label for y-axis.
    plt.legend()
    plt.title('Calinski-Harabasz score vs number of clusters')

    plt.tight_layout()  # Adjust layout for better visualization.
    save_path = save_with_incremental_filename('Plots/clustering_scores.png')  # Generate a unique file name for saving.
    plt.savefig(save_path)  # Save the plot as an image.
    plt.show()  # Display the plot.


def plot_histplot_percentile(dataset, threshold):
    """
    Plots a histogram of distances with a vertical line indicating the threshold.

    Parameters:
        dataset (list or np.ndarray): Distance values to be plotted.
        threshold (float): Threshold value to be marked in the plot.

    Returns:
        None
    """
    sns.histplot(dataset)  # Plot the histogram of distances.
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold Distance: ({threshold:.2f})')  # Mark the threshold line.
    plt.xlabel('Distance')  # Label for x-axis.
    plt.ylabel('Frequency')  # Label for y-axis.
    plt.legend()
    plt.title('Distribution of distances to cluster centers')
    save_path = save_with_incremental_filename('Plots/clustering_distances.png')  # Generate a unique file name for saving.
    plt.savefig(save_path)  # Save the plot as an image.
    plt.show()  # Display the plot.


def plot_actor_presence(count_df, value_col, cluster_choice, title):
    """
    Plots a histogram of actor presence across clusters.

    Parameters:
        count_df (pd.DataFrame): DataFrame containing cluster counts.
        value_col (str): Column name specifying cluster information.
        cluster_choice (int): Number of clusters plus outliers.
        title (str): Title for the plot.

    Returns:
        None
    """
    sns.histplot(data=count_df, x=value_col, bins=cluster_choice + 1, discrete=True)  # Plot actor presence histogram.
    plt.title(f'Presence Count of Actors in {title}')  # Set the plot title.
    save_path = save_with_incremental_filename(f'Plots/Presence in {title} dataset.png')  # Generate a unique file name for saving.
    plt.savefig(save_path)  # Save the plot as an image.
    plt.xticks(ticks=range(-1, cluster_choice), labels=['outlier'] + list(range(cluster_choice)))  # Customize x-axis labels to include outliers.
    plt.show()  # Display the plot.


def plot_cluster_presence(cluster_dataframe):
    """
    Plots a heatmap of cluster presence across frames.

    Parameters:
        cluster_dataframe (pd.DataFrame): DataFrame indicating cluster presence in video frames.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))  # Create a figure for the heatmap.
    sns.heatmap(cluster_dataframe, cmap="YlGnBu", cbar=False, linewidths=0.5)  # Generate the heatmap.
    plt.title("Cluster Presence Across Frames")  # Set the plot title.
    plt.xlabel("Frame")  # Label for x-axis.
    plt.ylabel("Cluster")  # Label for y-axis.
    row_labels = cluster_dataframe.index.tolist()  # Extract cluster labels.
    row_labels = ['outlier' if label == -1 else label for label in row_labels]  # Replace -1 with 'outlier' in labels.
    plt.yticks(ticks=[i + 0.5 for i in range(len(row_labels))], labels=row_labels, rotation=0)  # Customize y-axis labels for readability.
    save_path = save_with_incremental_filename('Plots/Heatmap.png')  # Generate a unique file name for saving.
    plt.savefig(save_path)  # Save the heatmap as an image.
    plt.show()  # Display the heatmap.
