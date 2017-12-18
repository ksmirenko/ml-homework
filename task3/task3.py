from PIL import Image
import numpy as np

# Works when launched from terminal
# noinspection PyUnresolvedReferences
from k_means import k_means

input_image_file = 'lena.jpg'
output_image_prefix = 'out_lena'
n_clusters = [2, 3, 5]
max_iterations = 100
launch_count = 3


def main():
    # Read input image
    image = np.array(Image.open(input_image_file))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    for k in n_clusters:
        print(f"{k} clusters")
        # 'Compress' image using K-means
        centroids, clustered = k_means(X, k=k, max_iterations=max_iterations, launch_count=launch_count)
        new_X = np.array([centroids[cluster_index] for cluster_index in clustered])
        new_X = new_X.astype(np.uint8)

        # Write output image
        new_image = new_X.reshape(image.shape)
        output_image_name = f"{output_image_prefix}_{k}.jpg"
        Image.fromarray(new_image).save(output_image_name)
        print(f"Saved {output_image_name}")

    print("Done.")


main()
