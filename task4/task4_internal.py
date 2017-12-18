import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# For launching in IDEA
# from task3.k_means import k_means
# prefix = 'task4/'

# For launching from terminal
sys.path.append('../task3/')
prefix = ""
# noinspection PyUnresolvedReferences
from k_means import k_means

image_name = f"{prefix}policemen.jpg"
data_path = f"{prefix}policemen/"
output_image_path = f"{prefix}out_policemen.jpg"

# Configuration
n_clusters = [2, 3, 4, 5, 6, 7]
max_iterations = 100
launch_count = 3


def launch_k_means():
    image = np.array(Image.open(image_name))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    for k in n_clusters:
        print(f"{k} clusters")
        # 'Compress' image using K-means
        centroids, clustered = k_means(X, k=k, max_iterations=max_iterations, launch_count=launch_count)

        # Save the result for this k to file
        np.savetxt(f"{data_path}centroids_{k}.txt", X=centroids, delimiter='\t')
        clustered_compressed = np.array(clustered).astype(np.uint8)
        np.savetxt(f"{data_path}clustered_{k}.txt", X=clustered_compressed, delimiter='\t', fmt='%1d')

    print("Done clustering.")


# Load previously calculated clustering data for policemen.jpg
def load_clustered_data():
    image = np.array(Image.open(image_name))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    data = dict()
    for k in n_clusters:
        centroids = np.loadtxt(f"{data_path}centroids_{k}.txt", dtype=np.float64, delimiter='\t')
        clusters = np.loadtxt(f"{data_path}clustered_{k}.txt", dtype=np.uint8, delimiter='\t')
        data[k] = {'centroids': centroids, 'clusters': clusters}

    return X, data


def sq_dist(x1, x2):
    return sum([v ** 2 for v in (x1 - x2)])


def calculate_davies_bouldin_index(X, k, clusters, centroids):
    def sigma(i):
        cluster_points = [X[j] for j in range(X.shape[0]) if clusters[j] == i]
        return np.sqrt(np.sum([sq_dist(centroids[i], point) ** 2 for point in cluster_points]) / len(cluster_points))

    def db_index_for_pair(i, j):
        return (sigma(i) + sigma(j)) / sq_dist(centroids[i], centroids[j])

    def max_db_index_fof_cluster(i):
        return np.max([db_index_for_pair(i, j) for j in range(k) if i != j])

    return np.mean([max_db_index_fof_cluster(i) for i in range(k)])


def calculate_calinski_harabasz_score(X, k, clusters, centroids):
    cluster_points = [[X[j] for j in range(X.shape[0]) if clusters[j] == i] for i in range(k)]

    def s_w_for_cluster(i):
        # Suppressed warning: it's alright, cluster_points[i] is an array
        # noinspection PyTypeChecker
        return np.sum([sq_dist(point, centroids[i]) for point in cluster_points[i]])

    n = len(X)
    total_mean = np.mean(X, axis=0)
    s_w = np.sum([s_w_for_cluster(i) for i in range(k)])
    s_b = np.sum([(len(cluster_points[i]) * sq_dist(centroids[i], total_mean)) for i in range(k)])
    print(f"{s_w}\t{s_b}")
    return ((n - k) * s_b) / ((k - 1) * s_w)


def compress(centroids, clustered):
    image = np.array(Image.open(image_name))
    new_X = np.array([centroids[cluster_index] for cluster_index in clustered])
    new_X = new_X.astype(np.uint8)

    # Write output image
    new_image = new_X.reshape(image.shape)
    output_image_name = output_image_path
    Image.fromarray(new_image).save(output_image_name)
    print(f"Saved {output_image_name}")


# Perform cluster analysis
# This is a lengthy operation, so it can be done once and commented out later, since the output is stored in files
# launch_k_means()

# Calculate metrics
X, cluster_data = load_clustered_data()
db_scores = [calculate_davies_bouldin_index(X, k, cluster_data[k]['clusters'], cluster_data[k]['centroids'])
             for k in n_clusters]
ch_scores = [calculate_calinski_harabasz_score(X, k, cluster_data[k]['clusters'], cluster_data[k]['centroids'])
             for k in n_clusters]
# Save calculated metrics
np.savetxt(f"{prefix}policemen/db.txt", X=db_scores, delimiter='\t')
np.savetxt(f"{prefix}policemen/ch.txt", X=ch_scores, delimiter='\t')

# Draw plots
plt.plot(n_clusters, db_scores, color='blue', linestyle=':', linewidth=1, marker='s', markersize=6,
         markeredgecolor='black', markerfacecolor='blue', markeredgewidth=1,
         label=u'Davies-Bouldin index')
plt.xticks(np.arange(min(n_clusters), max(n_clusters) + 1, 1.0))
plt.xlabel('# of clusters')
plt.legend(loc='best')
plt.show()
plt.plot(n_clusters, ch_scores, color='red', linestyle=':', linewidth=1, marker='o', markersize=6,
         markeredgecolor='black', markerfacecolor='red', markeredgewidth=1,
         label=u'Calinski-Harabasz score')
plt.xticks(np.arange(min(n_clusters), max(n_clusters) + 1, 1.0))
plt.xlabel('# of clusters')
plt.legend(loc='best')
plt.show()

# Decide the best number of clusters
# Suppressed warnings below: np.argmin/argmax returns a scalar value here
# noinspection PyTypeChecker
best_k_db = n_clusters[np.argmin(db_scores)]
# noinspection PyTypeChecker
best_k_ch = n_clusters[np.argmax(ch_scores)]
print(f"Best # of clusters, based on Davies-Bouldin index, is {best_k_db}")
print(f"Best # of clusters, based on Calinski-Harabasz score, is {best_k_ch}")

# 'Compress' image using K-means
compress(centroids=cluster_data[best_k_db]['centroids'], clustered=cluster_data[best_k_db]['clusters'])
