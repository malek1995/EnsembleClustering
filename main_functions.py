from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

'''
# Function to create KMeans models for clustering data
#
# k_list: List of values for the number of clusters
# itr_num: Number of models to create for each value of k
# data: Data to be clustered into k clusters
#
# Returns a list of KMeans models (not the labels) 
'''


def create_kmeans_models(k_list, itr_num, data):
    kmeans_models = []
    for k in k_list:
        for i in range(itr_num):
            model = KMeans(n_clusters=k)
            model.fit(data)
            kmeans_models.append(model)
    return kmeans_models


'''
# Function to create SpectralClustering models for clustering data
#
# k_list: List of values for the number of clusters
# itr_num: Number of models to create for each value of k
# data: Data to be clustered into k clusters
#
# Returns a list of SpectralClustering models
'''


def create_spectral_clustering_models(k_list, itr_num, data):
    spectral_clustering_models = []
    for k in k_list:
        for i in range(itr_num):
            model = SpectralClustering(n_clusters=k, assign_labels='discretize')
            model.fit(data)
            spectral_clustering_models.append(model)
    return spectral_clustering_models


'''
Function to add a Species column to a dataframe based on a target column
df: Dataframe to add the Species column to
target_column: Column used to determine the values in the Species column
Adds a new column 'Species' to the input dataframe where the values are formatted 
as 'cluster_' followed by the value of the target column. 
'''


def define_species(df, target_column):
    species = ["cluster_" + str(val) for val in target_column]
    df['Species'] = species


'''
Function to calculate external validation metrics for clustering results
true_labels: Array of true labels for the data
predicted_labels: Array of predicted labels for the data
Returns a list of external validation metrics including normalized mutual information score, 
adjusted Rand score, and Jaccard score
'''


def calculate_external_validation(true_labels, predicted_labels):
    nmf_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='geometric')
    ar_score = adjusted_rand_score(true_labels, predicted_labels)
    j_score = jaccard_score(true_labels, predicted_labels, average='weighted')
    return [round(nmf_score, 3), round(ar_score, 3), round(j_score, 3)]


'''
Function to calculate internal validation metrics for a given clustering model
actual_data: The original data before reduction or clustering
predicted_labels: The predicted cluster labels for the original data
Returns a list of internal validation metrics (Davies-Bouldin score, silhouette score, and Calinski-Harabasz score)
'''


def calculate_internal_validation(actual_data, predicted_labels):
    reduced_data = PCA(n_components=2).fit_transform(actual_data)
    db_score = davies_bouldin_score(reduced_data, predicted_labels)
    sh_score = silhouette_score(reduced_data, predicted_labels)
    ch_score = calinski_harabasz_score(reduced_data, predicted_labels)
    return [round(db_score, 3), round(sh_score, 3), round(ch_score, 3)]


def calculate_mean_of_validation(validation_results):
    first, second, third, length = 0, 0, 0, 0
    for [x, y, z] in validation_results:
        first += x
        second += y
        third += z
        length += 1
    return [round(first/length, 3), round(second/length, 3), round(third/length, 3)]
# %%
