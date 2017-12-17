import numpy as np
from PIL import Image

# For launching in IDEA to work properly
prefix = "task4/"
# prefix = ""
image_name = f"{prefix}policemen.jpg"


# Load previously calculated clustering data for policemen.jpg
def load_clustered_data():
    image = np.array(Image.open(image_name))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    data = dict()
    for k in range(2, 8):
        centroids = np.loadtxt(f"{prefix}policemen/centroids_{k}.txt", dtype=np.float64, delimiter='\t')
        clusters = np.loadtxt(f"{prefix}policemen/clustered_{k}.txt", dtype=np.uint8, delimiter='\t')
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


X, cluster_data = load_clustered_data()

# db_scores = [calculate_davies_bouldin_index(X, k, cluster_data[k]['clusters'], cluster_data[k]['centroids'])
#              for k in range(2, 8)]

# ch_scores = [calculate_calinski_harabasz_score(X, k, cluster_data[k]['clusters'], cluster_data[k]['centroids'])
#              for k in range(2, 8)]

# np.savetxt(f"{prefix}policemen/ch.txt", X=ch_scores, delimiter='\t')
