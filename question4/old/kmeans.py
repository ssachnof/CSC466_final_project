import sys
import pandas as pd
import numpy as np

def euclidean_distance(df, x):
    return np.sqrt(np.sum(np.square(df - x), axis=1))

def main():
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print('Usage: python3 kmeans.py <Filename> <k>')
		sys.exit()

	filename, k = sys.argv[1], int(sys.argv[2])
	df = pd.read_csv(filename).drop('0', axis=1, errors='ignore')

	# Compute initial centroids
	D = df.apply(lambda x : euclidean_distance(df, x), axis=1)
	max_point = D.max().idxmax()
	initial_centroids = [max_point, D.idxmax()[max_point]]
	for _ in range(k-2):
	    next_centroid = D[initial_centroids].drop(initial_centroids).sum(axis=1).idxmax()
	    initial_centroids.append(next_centroid)
	df['cluster'] = D[initial_centroids].idxmin(axis=1)
	centroids = df.groupby('cluster').mean()
	centroids.index = range(k)

	# Assign Points to inital Centroids
	point_centroid_dists = df.apply(lambda x : euclidean_distance(centroids, x), axis=1)
	SSE = point_centroid_dists.min(axis=1).sum()
	df['cluster'] = point_centroid_dists.idxmin(axis=1)

	# Perform K-Means Clustering
	stop_criteria = False
	prev_centroids = None
	prev_point_assignment = None
	prev_SSE = 0
	SSE_thresh = 0.01
	while not stop_criteria:
	    # Save current values
	    prev_centroids = centroids.copy()
	    prev_point_assignment = df['cluster'].copy()
	    SSE_prev = SSE
	    
	    # Recalculate centroids
	    centroids = df.groupby('cluster').mean()
	    point_centroid_dists = df.apply(lambda x : euclidean_distance(centroids, x), axis=1)
	    df['cluster'] = point_centroid_dists.idxmin(axis=1)
	    
	    # Check Stopping Conditions
	    no_point_reassign = df['cluster'].equals(prev_point_assignment)
	    no_centroid_reassign =  centroids.equals(prev_centroids)
	    SSE = point_centroid_dists.min(axis=1).sum()
	    no_SSE_decrease = (SSE - prev_SSE) / SSE < SSE_thresh
	    stop_criteria = no_point_reassign or no_centroid_reassign or no_SSE_decrease

	# Print Results
	for centroid, center in centroids.iterrows():
	    cluster_points = df.cluster[df.cluster==centroid].index.values
	    cluster_dists = point_centroid_dists[centroid][cluster_points]
	    print('Cluster', centroid)
	    print('Center', center.values)
	    print('Max Dist. to Center:', cluster_dists.max())
	    print('Min Dist. to Center:', cluster_dists.min())
	    print('Avg Dist. to Center:', cluster_dists.mean())
	    print('Sum of Squared Errors:', SSE)
	    print(len(cluster_points), 'Points')
	    print(df[df.cluster==centroid].drop('cluster', axis=1))
	    print()

if __name__ == "__main__":
	main()
