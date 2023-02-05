import numpy as np
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import ClusterEnsembles as CE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# The following vars are for displaying the end results for each algorithm. See display_results in main_function

list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list = \
    [], [], [], [], []


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
Function to create ensemble clustering labels for clustering data
k_list: List of values for the number of clusters
itr_num: Number of models to create for each value of k
ensemble_input: Data to be clustered into k clusters using ensemble clustering
algo_name: name of the algorithm to use for ensemble clustering (must be one of cspa, mcla, hbgf, nmf)
Returns a list of ensemble clustering labels with the specified number of clusters and algorithm
'''


def create_ensemble_clustering(k_list, itr_num, ensemble_input, algo_name):
    ensemble_methods = ('cspa', 'mcla', 'hbgf', 'nmf')
    assert algo_name.lower() in ensemble_methods, 'The given algorithm name is not in [cspa, mcla, hbgf, nmf], ' \
                                                  'please use one of these algorithms'
    ensemble_labels = []
    for k in k_list:
        for i in range(itr_num):
            label = CE.cluster_ensembles(ensemble_input, nclass=k, solver=algo_name)
            ensemble_labels.append(label)
    return ensemble_labels


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


'''
Function to calculate the mean of the validation scores
validation_results: List of results obtained from calculating internal or external validation
Returns a list of the mean values for each validation score
'''


def calculate_mean_of_validation(validation_results):
    first_score, second_score, third_score, length = 0, 0, 0, 0
    for [x, y, z] in validation_results:
        first_score += x  # x is either davies_bouldin_score or  normalized_mutual_info_score
        second_score += y  # y is either silhouette_score or adjusted_rand_score
        third_score += z  # z is either calinski_harabasz_score or jaccard_score
        length += 1  # length is the size of validation_results
    return [round(first_score / length, 3), round(second_score / length, 3), round(third_score / length, 3)]


'''
Function to extract the predicted labels from a list of models
models: List of clustering models
Returns a list of predicted labels, one for each model in the input list
'''


def get_labels_from_models(models):
    return [model.labels_ for model in models]


'''
Function to convert list of clustering models into valid input for ensemble clustering
models: List of clustering models
Returns an array of predicted labels from the clustering models, suitable for input into ensemble clustering methods.
'''


def modify_models_to_valid_ensemble_input(models):
    return np.array(get_labels_from_models(models))


'''
Function to calculate internal validation metrics for multiple labels
actual_data: Original data to compare against predicted clusters
predicted_labels: List of predicted labels generated from different models
Returns a list of internal validation metrics for each model in the form of [db_score, sh_score, ch_score]
'''


def calculate_internal_validation_for_all_labels(actual_data, predicted_labels):
    validation_results = []
    for label in predicted_labels:
        validation_results.append(calculate_internal_validation(actual_data, label))
    return validation_results


'''
Function to calculate external validation scores for a list of predicted labels
true_labels: Ground truth labels for the data
predicted_labels: List of predicted labels obtained from different models
Returns a list of lists, each inner list contains the external validation scores for one set of predicted labels
'''


def calculate_external_validation_for_all_labels(true_labels, predicted_labels):
    validation_results = []
    for label in predicted_labels:
        validation_results.append(calculate_external_validation(true_labels, label))
    return validation_results


'''
Calculates the variance of the validation scores in the validation_results list.    
Parameters:
validation_results (List): List of results obtained from calculating internal or external validation   
mean (List): List of mean values for the given validation_results 
Returns:
List: A list of the variance values for each validation score
'''


def calculate_variance_of_validation(validation_results, mean):
    first_sum_square, second_sum_square, third_sum_square, length = 0, 0, 0, 0
    for [x, y, z] in validation_results:
        first_sum_square += (x - mean[0]) ** 2
        second_sum_square += (y - mean[1]) ** 2
        third_sum_square += (z - mean[2]) ** 2
        length += 1
    return [round(first_sum_square / length, 3), round(second_sum_square / length, 3),
            round(third_sum_square / length, 3)]


"""
This function use the following - list of algorithm names, list of k values, number of iterations, 
lists of mean values for internal and external validation, and lists of variance values for internal 
and external validation.
The function returns a table with the input values organized in a readable format, displaying the algorithm names, 
k values, iteration number, and mean and variance values for both internal and external validation.
"""


def display_results(k_list, itr_num):
    x = PrettyTable()
    x.field_names = ["", *list_of_algos]
    x.add_row(["k_list", *k_list * len(list_of_algos)])
    x.add_row(["itr_num", *[itr_num] * len(list_of_algos)])
    internal_mean_fields = ["Internal_Mean_DB_Score", "Internal_Mean_Sil_Score", "Internal_Mean_CH_Score"]
    for i in range(3):
        row = [internal_mean_fields[i]]
        for j in range(len(list_of_algos)):
            row.append(internal_mean_list[j][i])
        x.add_row(row)
    external_mean_fields = ["External_Mean_NMI_Score", "External_Mean_ARI_Score", "External_Mean_J_Score"]
    for i in range(3):
        row = [external_mean_fields[i]]
        for j in range(len(list_of_algos)):
            row.append(external_mean_list[j][i])
        x.add_row(row)
    internal_variance_fields = ["Internal_Variance_DB_Score", "Internal_Variance_Sil_Score",
                                "Internal_Variance_CH_Score"]
    for i in range(3):
        row = [internal_variance_fields[i]]
        for j in range(len(list_of_algos)):
            row.append(internal_variance_list[j][i])
        x.add_row(row)
    external_variance_fields = ["External_Variance_NMI_Score", "External_Variance_ARI_Score",
                                "External_Variance_J_Score"]
    for i in range(3):
        row = [external_variance_fields[i]]
        for j in range(len(list_of_algos)):
            row.append(external_variance_list[j][i])
        x.add_row(row)
    print(x)


"""
Adds the results of a single algorithm to the lists of results.
    
Args:
    - algo_name (str): name of the algorithm
    - internal_mean (list): list of internal mean scores for each evaluation metric
    - external_mean (list): list of external mean scores for each evaluation metric
    - internal_variance (list): list of internal variance scores for each evaluation metric
    - external_variance (list): list of external variance scores for each evaluation metric
    
Returns:
    None (updates the global lists)
"""


def add_results(algo_name, internal_mean, external_mean, internal_variance, external_variance):
    list_of_algos.append(algo_name)
    internal_mean_list.append(internal_mean)
    external_mean_list.append(external_mean)
    internal_variance_list.append(internal_variance)
    external_variance_list.append(external_variance)


def clear_results():
    global list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list
    list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list = \
        [], [], [], [], []

# %%
