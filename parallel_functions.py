import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import ClusterEnsembles as CE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import struct
import time
import multiprocessing as mp
from sklearn.datasets import load_iris

# The following vars are for displaying the end results for each algorithm. See display_results in main_function

list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list, \
    internal_min_list, external_min_list, internal_max_list, external_max_list = \
    [], [], [], [], [], [], [], [], []

df = pd.DataFrame()

time_dic = mp.Manager().dict()


def perform_ensemble_clustering(base_clustering, k_list, itr_num, data, target):
    ensemble_methods = ['nmf', 'cspa', 'mcla', 'hbgf']
    for method in ensemble_methods:
        print(f'Start run perform_ensemble_clustering on {method}')
        ensemble_labels = create_ensemble_clustering(k_list, itr_num, base_clustering, method, data)
        print(f'Done run perform_ensemble_clustering on {method}')
        add_validation_results(data, target, ensemble_labels, method)
        



def create_ensemble_clustering(k_list, itr_num, base_clustering, algo_name, data):
    
    ensemble_methods = ('cspa', 'mcla', 'hbgf', 'nmf')
    assert algo_name.lower() in ensemble_methods, 'The given algorithm name is not in [cspa, mcla, hbgf, nmf], ' \
                                                  'please use one of these algorithms'
    ensemble_labels = []
    queue = mp.Queue()
    for k in k_list:
        procs = []
        for i in range(itr_num):
            proc = mp.Process(target= run_ensemble, args=(queue, k_list, itr_num, k, base_clustering, algo_name, data)) 
            procs.append(proc)
            print("start porc", i)
            proc.start()    
        
        for i in range(itr_num):
            ensemble_labels.append(queue.get())
        
        for i in procs:
            i.join()
    return ensemble_labels
    
 


    

def run_ensemble(queue, k_list, itr_num, k, base_clustering, algo_name, data):
    global time_dic
    if base_clustering.lower() == 'kmeans':
        k_means_models = create_kmeans_models(k_list, itr_num, data)
        ensemble_input = modify_models_to_valid_ensemble_input(k_means_models)
    else:
        spectral_models = create_spectral_clustering_models(k_list, itr_num, data)
        ensemble_input = modify_models_to_valid_ensemble_input(spectral_models)
    
                              
    print(algo_name, "is running")
    start_time = time.time()
    label = CE.cluster_ensembles(ensemble_input, nclass=k, solver=algo_name)
    print(algo_name, "is done")
    end_time = time.time()
    time_taken = end_time - start_time
    if algo_name in time_dic:
        time_dic[algo_name].append(time_taken)
    else:
        time_dic[algo_name] = [time_taken] 
    queue.put(label)
    
    
    
def create_kmeans_models(k_list, itr_num, data):
    kmeans_models = []
    for k in k_list:
        for i in range(itr_num):
            model = KMeans(n_clusters=k, init='random', n_init = 1)
            model.fit(data)
            kmeans_models.append(model)
    return kmeans_models

















        
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


def calculate_internal_validation_for_all_labels(actual_data, predicted_labels, algo_name):
    validation_results = []
    for label in predicted_labels:
        validation_results.append(calculate_internal_validation(actual_data, label, algo_name))
    return validation_results



def calculate_external_validation_for_all_labels(true_labels, predicted_labels):
    validation_results = []
    for label in predicted_labels:
        validation_results.append(calculate_external_validation(true_labels, label))
    return validation_results


def calculate_external_validation(true_labels, predicted_labels):
    nmf_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='geometric')
    ar_score = adjusted_rand_score(true_labels, predicted_labels)
    j_score = jaccard_score(true_labels, predicted_labels, average='weighted')
    return [nmf_score, ar_score, j_score]




def calculate_internal_validation(actual_data, predicted_labels, algo_name):
    if algo_name.lower() == 'dbscan':
        reduced_data = actual_data
    else:
        reduced_data = PCA(n_components=2).fit_transform(actual_data)
    db_score = davies_bouldin_score(reduced_data, predicted_labels)
    sh_score = silhouette_score(reduced_data, predicted_labels)
    ch_score = calinski_harabasz_score(reduced_data, predicted_labels)
    return [db_score, sh_score, ch_score]


def get_labels_from_models(models):
    return [model.labels_ for model in models]


def modify_models_to_valid_ensemble_input(models):
    return np.array(get_labels_from_models(models))

def clear_results():
    global time_dic, df, list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list
    list_of_algos, internal_mean_list, external_mean_list, internal_variance_list, external_variance_list = \
        [], [], [], [], []
    df = pd.DataFrame()
    time_dic = {}








