import comonotonic as cm
import numpy as np
import pandas as pd
from multiprocessing import Pool
import operator
from collections import Counter

def single_ciber(bootstrap_x, bootstrap_y, discrete_feature_val, cont_col, categorical,
                 min_corr, cate_clusters, corrtype, discrete_method, sequence_num):
    ciber_clf = cm.clustered_comonotonic(bootstrap_x, bootstrap_y, discrete_feature_val, cont_col, categorical,
                                        min_corr, cate_clusters, corrtype, discrete_method)
    ciber_clf.run()
    return ciber_clf, sequence_num

def prob_dist_single(clf, x, sample_col):
    sample_x = x[sample_col]
    return clf.get_prob_dist_single(sample_x)

class ciber_forest:
    
    def __init__(self, n_estimators, max_workers, sample_percentage = 0.5):
        self.n_estimators = n_estimators
        self.max_workers = max_workers
        self.sample_percentage = sample_percentage
    
    def bootstrap_training(self, x_train, y_train):
        sample_size = x_train.shape[0]
        sample_row = list(np.random.choice(x_train.shape[0], sample_size))
        sample_row.sort()
        sample_dim = int(self.sample_percentage * x_train.shape[1])
        sample_col = list(np.random.choice(x_train.shape[1], sample_dim, replace = False))
        sample_col.sort()
        bootstrap_x = x_train[sample_row,:][:, sample_col]
        bootstrap_y = y_train[sample_row]
        return bootstrap_x, bootstrap_y, sample_col

    def fit(self, x_train, y_train, discrete_feature_val, cont_col, categorical,
            min_corr, cate_clusters, corrtype, discrete_method):
        param_collection = list()
        self.selected_col = dict()
        for itr in range(self.n_estimators):
            bootstrap_x, bootstrap_y, sample_col = self.bootstrap_training(x_train, y_train)
            
            column_check_book = dict() # key is the col index in the original x, value is the col index in bootstrap x
            for i, col_idx in enumerate(sample_col):
                column_check_book[col_idx] = i
            if discrete_feature_val != None:
                new_discrete_feature_val = dict()
                for k in discrete_feature_val.keys():
                    if k in column_check_book.keys():
                        new_discrete_feature_val[column_check_book[k]] = discrete_feature_val[k]
            else:
                new_discrete_feature_val = None
            
            new_cont_col = [column_check_book[col_idx] for col_idx in cont_col if col_idx in column_check_book.keys()]
            new_categorical = [column_check_book[col_idx] for col_idx in categorical if col_idx in column_check_book.keys()]
            
            if cate_clusters != None:
                new_cate_clusters = list()
                for cluster in cate_clusters:
                    new_cluster = list()
                    for col_idx in cluster:
                        if col_idx in column_check_book.keys():
                            new_cluster.append(column_check_book[col_idx])
                    if len(new_cluster) > 0:
                        new_cate_clusters.append(new_cluster)
            else:
                new_cate_clusters = None
            
            if new_discrete_feature_val != None and new_cate_clusters != None:
                params = (bootstrap_x.copy(), bootstrap_y.copy(), new_discrete_feature_val.copy(), new_cont_col.copy(), 
                        new_categorical.copy(), min_corr, new_cate_clusters.copy(), corrtype, discrete_method, itr)
            elif new_discrete_feature_val == None and new_cate_clusters != None:
                params = (bootstrap_x.copy(), bootstrap_y.copy(), None, new_cont_col.copy(), 
                        new_categorical.copy(), min_corr, new_cate_clusters.copy(), corrtype, discrete_method, itr)
            elif new_discrete_feature_val != None and new_cate_clusters == None:
                params = (bootstrap_x.copy(), bootstrap_y.copy(), new_discrete_feature_val.copy(), new_cont_col.copy(), 
                        new_categorical.copy(), min_corr, None, corrtype, discrete_method, itr)
            else:
                params = (bootstrap_x.copy(), bootstrap_y.copy(), None, new_cont_col.copy(), 
                        new_categorical.copy(), min_corr, None, corrtype, discrete_method, itr)
            param_collection.append(params)
            self.selected_col[itr] = sample_col.copy()
            del bootstrap_x, bootstrap_y, new_discrete_feature_val, new_cont_col, new_categorical, new_cate_clusters, column_check_book, sample_col
            
        pool = Pool(processes=self.max_workers)
        clf_collection = pool.starmap(single_ciber, param_collection)
        self.clf_collection = clf_collection
        pool.close()
    
    def ensemble_predict_single(self, x):
        param_collection = list()
        for clf, sequence_num in self.clf_collection:
            sample_col = self.selected_col[sequence_num]
            param_collection.append((clf, x, sample_col.copy()))
            del sample_col
        
        pool = Pool(processes=self.max_workers)
        prob_dist_collection = pool.starmap(prob_dist_single, param_collection)
        pool.close()
        
        prediction = list()
        ensembled_prob_dist = dict()
        for prob_dist in prob_dist_collection:
            predicted_label = max(prob_dist.items(), key = operator.itemgetter(1))[0]
            prediction.append(predicted_label)
            for k in prob_dist.keys():
                if k in ensembled_prob_dist.keys():
                    ensembled_prob_dist[k] += prob_dist[k]
                else:
                    ensembled_prob_dist[k] = prob_dist[k]
        freq = Counter(prediction).most_common()
        if len(freq) > 1 and freq[0][1] == freq[1][1]: # more than 1 predicted class
            return max(ensembled_prob_dist.items(), key=operator.itemgetter(1))[0]
        else:
            return freq[0][0]

    def predict(self, x_test):
        prediction = list()
        for x in x_test:
            result = self.ensemble_predict_single(x)
            prediction.append(result)
        return prediction