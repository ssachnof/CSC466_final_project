#CSC466 final project
#Samuel Erling Sachnoff (ssachnof@calpoly.edu)
#Ryan Holt (ryholt@calpoly.edu)
#Conor Whatley (cwhatley@calpoly.edu)
#Grayson Clendenon (gclenden@calpoly.edu)

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    if len(sys.argv) == 3:
        #in_file = sys.argv[1]
        out_file = sys.argv[1]
        k = int(sys.argv[2])
    else:
        print("Usage: python3 style_clustering.py out_file k")
        return

    in_file = "data/team_stats_2018.csv"

    drop_list = ["RK", "TOTAL_DVOA", "LAST_YEAR", "WEI_DVOA", "RANK", "W-L", "YEAR"]

    df = pd.read_csv(in_file).set_index("TEAM")
    df = df.iloc[1:]
    test = df.drop(drop_list, axis=1)
    
    clustering = kmeans(test,k)
    plot_clusters(df, test, clustering)

    stats = get_cluster_statistics(df, test, clustering)
    print_cluster_statistics(stats)

    with open(out_file, 'w') as out:
        write_clusters(clustering, k, out)
        write_cluster_statistics(stats, out)

    

def euclidean_distance(df, x):
    return np.sqrt(np.sum(np.square(df - x), axis=1))


def kmeans(df, k):
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

    clustering = []
    for team, row in df.iterrows():
        clustering += [(team, int(row["cluster"]))]
    
    return clustering    


def plot_clusters(df, test, clustering):
    x_labels = list(test.index.values)
    ranks = list(df["RK"])
    #x = list(range(len(x_labels)))
    x = list(test["ST_DVOA"])
    y = list(test["OFFENSE_DVOA"])
    z = list(test["DEFENSE_DVOA"])

    colors = []
    for _, cluster in clustering:
        colors += [cluster]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors)
    ax.set_xlabel("ST_DVOA")
    ax.set_ylabel("OFFENSE_DVOA")
    ax.set_zlabel("DEFENSE_DVOA")

    for i in range(len(ranks)):
        ax.text(x[i], y[i], z[i], ranks[i])

    plt.show()


def get_cluster_statistics(df, test, clustering):
    teams, clusters = map(list,zip(*clustering))
    test["cluster"] = clusters
    test["rank"] = df["RANK"]

    means = test.groupby("cluster").mean()
    #print(test.groupby("cluster").count()["rank"])
    means["count"] = test.groupby("cluster").count()["rank"]
    return means
def print_cluster_statistics(stats_df):
    out_cols = ["rank","OFF_RANK", "DEF_RANK", "ST_RANK"]
    for cluster, row in stats_df.iterrows():
        print("Cluster: {}".format(cluster))
        print("\tCount: {}".format(row["count"]))
        for col in out_cols:
            print("\tMEAN {}: {}".format(col, row[col]))

def write_clusters(clustering, k, out):
    out.write("Clustering Output (K-Means, k = {}):\n".format(k))
    for pair in clustering:
        out.write("\tTeam: {}\tCluster: {}\n".format(pair[0], pair[1]))

def write_cluster_statistics(stats_df, out):
    out.write("\nCluster Statistics:\n")
    out_cols = ["rank","OFF_RANK", "DEF_RANK", "ST_RANK"]
    for cluster, row in stats_df.iterrows():
        out.write("\tCluster: {}\n".format(cluster))
        out.write("\t\tCount: {}\n".format(row["count"]))
        for col in out_cols:
            out.write("\t\tMEAN {}: {}\n".format(col, row[col]))



if __name__ == '__main__':
    main()
