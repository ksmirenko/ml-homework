import sys
import numpy as np
import matplotlib.pyplot as plt

# For launching in IDEA
# from task3.k_means import k_means
# prefix = 'task4/'

# For launching from terminal
sys.path.append('../task3/')
prefix = ""
# noinspection PyUnresolvedReferences
from k_means import k_means

in_data_path = f"{prefix}task_2_data_7.txt"
out_data_path = f"{prefix}data_7/"
n_clusters = [2, 3, 4, 5]
max_iterations = 100
launch_count = 3


def count_rand_and_fm(expected_clusters, actual_clusters):
    n = len(actual_clusters)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if actual_clusters[i] == actual_clusters[j]:
                if expected_clusters[i] == expected_clusters[j]:
                    tp += 1
                else:
                    fp += 1
            else:
                if expected_clusters[i] != expected_clusters[j]:
                    tn += 1
                else:
                    fn += 1
    n_pairs = n * (n - 1) / 2

    rand = (tp + tn) / n_pairs
    fowlkes_mallows = tp / np.sqrt((tp + fp) * (tp + fn))
    return np.array([rand, fowlkes_mallows])


def launch_k_means(X):
    clusters = []
    for k in n_clusters:
        print(f"{k} clusters")
        # 'Compress' image using K-means
        _, clustered = k_means(X, k=k, max_iterations=max_iterations, launch_count=launch_count)
        clusters.append(clustered)

        # Save the result for this k to file
        # clustered_compressed = np.array(clustered).astype(np.uint8)
        # np.savetxt(f"{out_data_path}clustered_{k}.txt", X=clustered_compressed, delimiter='\t', fmt='%1d')

    print("Done clustering.")
    return np.array(clusters)


# Perform cluster analysis and calculate metrics
data = np.genfromtxt(in_data_path, delimiter=' ')
y, X = np.hsplit(data, [1])
clusters = launch_k_means(X)
scores = np.array([count_rand_and_fm(expected_clusters=y, actual_clusters=clustered) for clustered in clusters])

# Draw plots
rand_plot, = plt.plot(n_clusters, scores[:, 0], color='blue', linestyle=':', linewidth=1, marker='s', markersize=6,
                      markeredgecolor='black', markerfacecolor='blue', markeredgewidth=1, label=u'Rand')
fm_plot, = plt.plot(n_clusters, scores[:, 1], color='red', linestyle=':', linewidth=1, marker='o', markersize=6,
                    markeredgecolor='black', markerfacecolor='red', markeredgewidth=1, label=u'Fowlkes-Mallows')
plt.xticks(np.arange(min(n_clusters), max(n_clusters) + 1, 1.0))
plt.xlabel('# of clusters')
plt.legend(loc='best')
plt.show()

# Decide the best number of clusters
best_n_of_clusters = [n_clusters[i] for i in np.argmax(scores, axis=0)]
print(f"Best # of clusters, based on Rand metric, is {best_n_of_clusters[0]}")
print(f"Best # of clusters, based on Fowlkes-Mallows metric, is {best_n_of_clusters[0]}")
