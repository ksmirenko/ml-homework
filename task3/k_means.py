import numpy as np
from numpy import random


def k_means(X, k, max_iterations, launch_count):
    def sq_dist(x1, x2):
        return sum([v ** 2 for v in (x1 - x2)])

    data_length = X.shape[0]

    best_functional = -1
    best_centroids = np.zeros(shape=data_length, dtype=np.float64)
    best_clustered = np.zeros(shape=data_length, dtype=np.int8)

    for launch_number in range(launch_count):
        print(f"\tLaunch #{launch_number}")
        # Initialization: pick random points as initial centroids
        centroids = X[random.choice(range(data_length), size=k, replace=False)].astype(np.float64)
        # Put all points into cluster #0
        clustered = np.zeros(shape=data_length, dtype=np.int8)

        n = 1  # iteration number
        converged = False
        while n < max_iterations and not converged:
            # Classification: recalculate cluster indices for each point
            clustered = [np.argmin([sq_dist(x, c) for c in centroids]) for x in X]
            # Minimization: recalculate centroids
            converged = True
            for i in range(k):
                cluster = np.array([X[j] for j in range(data_length) if clustered[j] == i])
                new_centroid = np.mean(cluster, axis=0)
                if sq_dist(centroids[i], new_centroid) > 1e-6:
                    centroids[i] = new_centroid
                    converged = False

        # Mitigate the local min problem: use the outputs with the least functional value
        target_functional = 0.
        for i in range(k):
            c = centroids[i]
            target_functional += sum([sq_dist(c, X[j]) for j in range(data_length) if clustered[j] == i])
        print(f"\t\tTarget functional = {target_functional}")
        if best_functional > -1:
            if target_functional < best_functional:
                print("\t\tFunctional improved; saved the current results as best.")
                best_functional = target_functional
                best_centroids = centroids
                best_clustered = clustered
        else:
            print("\t\tThis is the first launch; saved the current results as best.")
            best_functional = target_functional
            best_centroids = centroids
            best_clustered = clustered

    return best_centroids, best_clustered
