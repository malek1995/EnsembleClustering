import numpy as np
import pandas as pd
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
import struct
from sklearn.cluster import DBSCAN
import time

# The following vars are for displaying the end results for each algorithm. See display_results in main_function

list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list, \
    internal_min_list, external_min_list, internal_max_list, external_max_list = \
    [], [], [], [], [], [], [], [], []

df = pd.DataFrame()

time_dic = {}

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
            model = KMeans(n_clusters=k, init='random', n_init = 1)
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


def create_dbscan_clustering_models(eps_list, min_samples, data):
    dbscan_models = []
    for eps in eps_list:
        for min_sample in min_samples:
            model = DBSCAN(eps, min_sample)
            model.fit(data)
            dbscan_models.append(model)
    return dbscan_models
            
'''
Function to add a Species column to a dataframe based on a target column
df: Dataframe to add the Species column to
target_column: Column used to determine the values in the Species column
Adds a new column 'Species' to the input dataframe where the values are formatted 
as 'cluster_' followed by the value of the target column. 
'''


# def define_species(df, target_column):
#     species = ["cluster_" + str(val) for val in target_column]
#     df['Species'] = species


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
    return [nmf_score, ar_score, j_score]


'''
Function to calculate internal validation metrics for a given clustering model
actual_data: The original data before reduction or clustering
predicted_labels: The predicted cluster labels for the original data
Returns a list of internal validation metrics (Davies-Bouldin score, silhouette score, and Calinski-Harabasz score)
'''


def calculate_internal_validation(actual_data, predicted_labels, algo_name):
    if algo_name.lower() == 'dbscan':
        reduced_data = actual_data
    else:
        reduced_data = PCA(n_components=2).fit_transform(actual_data)
    db_score = davies_bouldin_score(reduced_data, predicted_labels)
    sh_score = silhouette_score(reduced_data, predicted_labels)
    ch_score = calinski_harabasz_score(reduced_data, predicted_labels)
    return [db_score, sh_score, ch_score]




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


def calculate_internal_validation_for_all_labels(actual_data, predicted_labels, algo_name):
    validation_results = []
    for label in predicted_labels:
        validation_results.append(calculate_internal_validation(actual_data, label, algo_name))
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










def clear_results():
    global time_dic, df, list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list
    list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list = \
        [], [], [], [], []
    df = pd.DataFrame()
    time_dic = {}

"""
This function performs the internal and external validation calculation for a given clustering algorithm and 
stores the results (mean and variance) in the corresponding lists. 
The results are stored using the add_results function.
Inputs:
    data: The data to be clustered.
    target: The true target values for the data.
    labels: The labels generated by the clustering algorithm.
    algo_name: The name of the clustering algorithm to be used in the results table.
Outputs:
    None. The results are stored in the corresponding lists.
"""


def add_validation_results(data, target, labels, algo_name):
    global df
    
    internal = calculate_internal_validation_for_all_labels(data, labels, algo_name)
    external = calculate_external_validation_for_all_labels(target, labels)

    internal_scores = list(zip(*internal))
    external_scores = list(zip(*external))
    
    db_score = list(internal_scores[0])
    sh_score = list(internal_scores[1])
    ch_score = list(internal_scores[2])
    nmf_score = list(external_scores[0])
    ar_score = list(external_scores[1])
    j_score = list(external_scores[2])


    df = df.append({'Algorithm': algo_name, 'DB_Score': db_score, 'SH_Score': sh_score, 'CH_Score': ch_score,
                   'NMF_Score': nmf_score, 'AR_Score': ar_score, 'J_Score': j_score}, ignore_index=True)

    
'''
Function for generating ensemble clustering models
Inputs:
base_models: list of base clustering models to use for ensemble clustering
k_list: list of number of clusters to generate for each iteration
itr_num: number of iterations to run the ensemble clustering
data: input data to be clustered
target: target labels for the input data
ensemble_methods: list of ensemble methods to use, options are ['NMF', 'CSPA', 'MCLA', 'HBGF']
Output: None (results are stored in global variables for validation purposes)
'''


def perform_ensemble_clustering(base_clustering, k_list, itr_num, data, target):
    ensemble_methods = ['nmf', 'cspa', 'mcla', 'hbgf']
    for method in ensemble_methods:
        ensemble_labels = create_ensemble_clustering(k_list, itr_num, base_clustering, method, data)
        add_validation_results(data, target, ensemble_labels, method)


'''
Function to create ensemble clustering labels for clustering data
k_list: List of values for the number of clusters
itr_num: Number of models to create for each value of k
ensemble_input: Data to be clustered into k clusters using ensemble clustering
algo_name: name of the algorithm to use for ensemble clustering (must be one of cspa, mcla, hbgf, nmf)
Returns a list of ensemble clustering labels with the specified number of clusters and algorithm
'''


def create_ensemble_clustering(k_list, itr_num, base_clustering, algo_name, data):
    global time_dic
    ensemble_methods = ('cspa', 'mcla', 'hbgf', 'nmf')
    assert algo_name.lower() in ensemble_methods, 'The given algorithm name is not in [cspa, mcla, hbgf, nmf], ' \
                                                  'please use one of these algorithms'
    ensemble_labels = []
    for k in k_list:
        for i in range(itr_num):
            if base_clustering.lower() == 'kmeans':
                k_means_models = create_kmeans_models(k_list, itr_num, data)
                ensemble_input = modify_models_to_valid_ensemble_input(k_means_models)
            else:
                spectral_models = create_spectral_clustering_models(k_list, itr_num, data)
                ensemble_input = modify_models_to_valid_ensemble_input(spectral_models)
            start_time = time.time()
            label = CE.cluster_ensembles(ensemble_input, nclass=k, solver=algo_name)
            end_time = time.time()
            time_taken = end_time - start_time
            
            if algo_name in time_dic:
                time_dic[algo_name].append(time_taken)
            else:
                time_dic[algo_name] = [time_taken]
            ensemble_labels.append(label)
    return ensemble_labels








def create_ensemble_clustering_with_dbscan(k_list, itr_num, dbscan_models, data, target):
    ensemble_methods = ['nmf', 'cspa', 'mcla', 'hbgf']
    ensemble_labels = []
    for method in ensemble_methods:
        for k in k_list:
            for i in range(itr_num):
                ensemble_input = modify_models_to_valid_ensemble_input(dbscan_models)
                label = CE.cluster_ensembles(ensemble_input, nclass=k, solver=method)
                ensemble_labels.append(label)
        add_validation_results(data, target, ensemble_labels, method)        
        ensemble_labels = []
    

    
    
    
    
    
    
    
        
def get_df():
    global df
    return df


def get_time_dic():
    global time_dic
    return time_dic





def display_results(df):
    validation_techniques = ['DB_Score', 'SH_Score', 'CH_Score', 'NMF_Score', 'AR_Score', 'J_Score']
    algo_names = list(df['Algorithm'])
    all_values = []
    for i in range(len(algo_names)):
        values_of_alog = []
        for j in validation_techniques:
            score = df.at[i, j]
            min_ = np.min(score)
            max_ = np.max(score)
            mean = np.mean(score)
            var = np.var(score)
            values_of_alog.extend([min_, max_, mean, var])
        all_values.append(values_of_alog)
        values_of_algo = []
    df = pd.DataFrame(all_values,
                          index=pd.Index(algo_names, name='Algorithm name'),
                          columns=pd.MultiIndex.from_product([validation_techniques,['Min', 'Max', 'Mean', 'Var']], names=['Validation technique:', '']))
    df = df.round(3)
    return df.style
  
    
    
    
    
    
    
    
    
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# %%
