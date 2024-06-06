import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from collections import defaultdict
import json
import os
from keras.utils import to_categorical
from tqdm import tqdm
from sklearn.utils import shuffle

import multiprocessing as mp

from functools import partial

import tensorflow as tf

"""
data loading and negative sampling
"""

class data_RW():
    def __init__(self):
        pass

    def json_load(self,path):
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return data

    def json_save(self,path,data):
        with open(path,'w',encoding='utf-8') as f:
            json.dump(data,f)

    def dataframe_load(self,path):
        data = pd.read_csv(path)
        return data

    def dataframe_save(self,path,data):
        data.to_csv(path,index=False)

    def txt_load(self,path):
        with open(path,'r',encoding='utf-8') as f:
            data = f.readlines()
        return data

class data_prepare():
    def __init__(self):
        pass

    def test_split(self,data,test_ratio=0.1,seed=42):
        train, test = train_test_split(data,test_size=test_ratio,random_state=seed)
        return train, test

    def get_sets(self,data,emo=False):
        if emo is False:
            u_id = data['user_id']
            i_id = data['song_id']
            ui = defaultdict(list)
            for i in range(len(data)):
                ui[u_id[i]].append(i_id[i])
            iids = set(i_id)
            return ui, iids, len(ui), len(iids)
        else:
            u_id = data['user_id']
            i_id = data['song_id']
            e_id = data['emo_id']
            ui = defaultdict(list)
            ue = defaultdict(list)
            for i in range(len(data)):
                ui[u_id[i]].append(i_id[i])
                ue[u_id[i]].append(e_id[i])
            iids = set(i_id)
            return ui, iids, len(ui), len(iids), ue

    def negative_sampling(self,ui,i_set,neg_num,seed=None):
        data_ = []
        if seed is None:
            for u_ in ui:
                u_iset = ui[u_]
                for i_ in u_iset:
                    item_j = random.sample(i_set-set(u_iset),neg_num)
                    data_.extend([[u_,i_,j_] for j_ in item_j])
        else:
            for u_ in ui:
                u_iset = ui[u_]
                for i_ in u_iset:
                    item_j = shuffle(list(i_set-set(u_iset)),random_state=seed)[:neg_num]
                    data_.extend([[u_,i_,j_] for j_ in item_j])
        return np.array(data_)

    def negative_sampling_emo(self,ui,i_set, ue,neg_num,seed=None,):
        data_ = []
        if seed is None:
            for u_ in ui:
                u_iset = ui[u_]
                u_eset = ue[u_]
                for i_num,i_ in enumerate(u_iset):
                    item_j = random.sample(i_set-set(u_iset),neg_num)
                    data_.extend([[u_,i_,j_,u_eset[i_num]] for j_ in item_j])
        else:
            for u_ in ui:
                u_iset = ui[u_]
                u_eset = ue[u_]
                for i_num, i_ in enumerate(u_iset):
                    item_j = shuffle(list(i_set-set(u_iset)),random_state=seed)[:neg_num]
                    data_.extend([[u_, i_, j_, u_eset[i_num]] for j_ in item_j])
        return np.array(data_)

    def neg_sampling_uipair(self,ui_pair,ui,i_set,neg_num, seed=None):
        data_ = []
        if seed is None:
            u_ = ui_pair[0]
            i_ = ui_pair[1]
            u_iset = ui[u_]
            items_j = random.sample(i_set-set(u_iset),neg_num)
            data_.extend([[u_,i_,j_] for j_ in items_j])
        else:
            u_ = ui_pair[0]
            i_ = ui_pair[1]
            u_iset = ui[u_]
            items_j = shuffle(list(i_set-set(u_iset)),random_state=seed)[:neg_num]
            data_.extend([[u_,i_,j_] for j_ in items_j])
        return data_

    def neg_sampling_mp(self,ui_pairs,ui,i_set,neg_num,process_num=8,seed=None):
        data = []
        if process_num>1:
            pool=mp.Pool(processes=process_num)
            res = pool.map(partial(self.neg_sampling_uipair,ui=ui,i_set=i_set,neg_num=neg_num),ui_pairs)
            pool.close()
            pool.join()
            data.extend([res_ for res_ in res])
        else:
            for ui_pair in ui_pairs:
                neg_samps = self.neg_sampling_uipair(ui_pair,ui,i_set,neg_num)
                data.extend(neg_samps)
        return np.array(data).reshape(-1,3)

    def neg_sampling_pointwise(self,ui, i_set, neg_num, seed=None):
        data_ = []
        if seed is None:
            for u_ in ui:
                u_iset = ui[u_]
                for i_ in u_iset:
                    data_.append([u_,i_,1])
                    item_j = random.sample(i_set-set(u_iset),neg_num)
                    data_.extend([[u_,j_,0] for j_ in item_j])
        return np.array(data_).reshape(-1,3)

    def neg_sampling_pointemo(self, ui, i_set, ue, neg_num, seed=None):
        data_ = []
        if seed is None:
            for u_ in ui:
                u_iset = ui[u_]
                u_eset = ue[u_]
                for i_num, i_ in enumerate(u_iset):
                    data_.append([u_, i_, u_eset[i_num], 1])
                    item_j = random.sample(i_set - set(u_iset), neg_num)
                    data_.extend([[u_, j_, u_eset[i_num], 0] for j_ in item_j])
        return np.array(data_).reshape(-1, 4)

class data4emo_select():
    def __init__(self):
        pass

    def csv_load(self,path):
        return pd.read_csv(path)

    def song_emo_load(self,path):
        return np.load(path)

    def user2song_emo(self,use,song_emo):
        user_emos = []
        song_emos = None
        for i in range(len(use)):
            user_emos.append(use['emo_id'][i])
            song_ = use['song_id'][i]
            if song_emos is None:
                song_emos = song_emo[song_].reshape(1,-1)
            else:
                song_emos = np.concatenate((song_emos,song_emo[song_].reshape(1,-1)),axis=0)
        return np.array(user_emos), song_emos


class Pretrained_Model():
    def __init__(self):
        pass

    def get_trained_params(self, path,model_name):
        trained_params = {}
        saver = tf.compat.v1.train.import_meta_graph(path+model_name+'.meta')#,clear_devices=True
        with tf.compat.v1.Session() as sess:
            saver.restore(sess,path+model_name)
            for var in tf.compat.v1.trainable_variables():
                trained_params[var.name] = sess.run(var)
        return trained_params
    def all_models(self):
        pre_Models = {}
        for cluster_n in range(50):
            PATH = "./pretrained_psi/"
            model_name = f"emo_select_model_g{cluster_n}"
            if cluster_n<10:
                model_path = PATH+f"0{cluster_n}/"
            else:
                model_path = PATH+f"{cluster_n}/"
            trained_PARAMS = self.get_trained_params(path=model_path,model_name=model_name)
            pre_Models[cluster_n] = trained_PARAMS
            tf.compat.v1.reset_default_graph()
        return pre_Models

    def reparameteriztion(self, mu, rho):
        epsilon = tf.compat.v1.random.normal(shape=tf.shape(rho), mean=0.0, stddev=1.0, seed=48)
        sigma = tf.compat.v1.log1p(tf.compat.v1.exp(rho))
        reparams = mu + epsilon * sigma
        return reparams