import numpy as np

from functools import partial
import multiprocessing as mp
import tensorflow as tf

def score(eval_data, eval_length=None):
    pass

def evaluate_one_ui(ui_pair,UI_score,ui_train,top_K_list=[10]):
    HR = {}
    NDCG = {}
    u = ui_pair[0]
    target_i = ui_pair[1]
    scores = UI_score[u]
    indexes = np.array([i for i in range(len(scores))])
    train_u_items = list(set(ui_train[u]))
    if target_i in train_u_items:
        train_u_items.remove(target_i)
    if train_u_items == []:
        pass
    else:
        scores = np.delete(scores, train_u_items)
        indexes = np.delete(indexes, train_u_items)
    score_sort = sorted([(indexes[i], scores[i]) for i in range(len(scores))], key=lambda x: x[1], reverse=True)
    rec_list = [item[0] for item in score_sort]
    for top_K in top_K_list:
        rec_list_K = rec_list[:top_K]
        hr = hit_rate(target_i, rec_list_K)
        ndcg = ndcg_(target_i, rec_list_K)
        HR[f"{top_K}"]=hr
        NDCG[f"{top_K}"]=ndcg
    return HR, NDCG

def evaluate_one_ui_emo(uie_pairs,UI_score,ui_train,top_K_list=[10]):
    HR = {}
    NDCG = {}
    for i_, ui_pair in enumerate(uie_pairs):
        u = ui_pair[0]
        target_i = ui_pair[1]
        scores = UI_score[i_]
        indexes = np.array([i for i in range(len(scores))])
        train_u_items = list(set(ui_train[u]))
        if target_i in train_u_items:
            train_u_items.remove(target_i)
        if train_u_items == []:
            pass
        else:
            scores = np.delete(scores, train_u_items)
            indexes = np.delete(indexes, train_u_items)
        score_sort = sorted([(indexes[i], scores[i]) for i in range(len(scores))], key=lambda x: x[1], reverse=True)
        rec_list = [item[0] for item in score_sort]
        for top_K in top_K_list:
            rec_list_K = rec_list[:top_K]
            hr = hit_rate(target_i, rec_list_K)
            ndcg = ndcg_(target_i, rec_list_K)
            if f"{top_K}" in HR:
                HR[f"{top_K}"].append(hr)
            else:
                HR[f"{top_K}"] = [hr]
            if f"{top_K}" in NDCG:
                NDCG[f"{top_K}"].append(ndcg)
            else:
                NDCG[f"{top_K}"] = [ndcg]
    return HR, NDCG

def evaluate_all(eval_ui,UI_score,ui_train,top_K_list=[10],process_num=8):
    HR_ = {}
    NDCG_ = {}
    for top_K in top_K_list:
        HR_[f"{top_K}"] = []
        NDCG_[f"{top_K}"] = []
    if process_num>1:
        pool = mp.Pool(processes=process_num)
        res = pool.map(partial(evaluate_one_ui,UI_score=UI_score,ui_train=ui_train,top_K_list=top_K_list),eval_ui)
        pool.close()
        pool.join()
        for res_ in res:
            for top_k in top_K_list:
                HR_[f"{top_k}"].append(res_[0][f"{top_k}"])
                NDCG_[f"{top_k}"].append(res_[1][f"{top_k}"])
    else:
        for ui_pair in eval_ui:
            hr, ndcg = evaluate_one_ui(ui_pair,UI_score,ui_train,top_K_list)
            for top_k in top_K_list:
                HR_[f"{top_k}"].append(hr[f"{top_k}"])
                NDCG_[f"{top_k}"].append(ndcg[f"{top_k}"])
    return HR_, NDCG_

def evaluate_all_emo(eval_uies, UI_score, ui_train, top_K_list=[10],process_num=8):
    HR_ = {}
    NDCG_ = {}
    for top_K in top_K_list:
        HR_[f"{top_K}"] = []
        NDCG_[f"{top_K}"] = []
    if process_num>1:
        pool = mp.Pool(processes=process_num)
        res = pool.map(partial(evaluate_one_ui_emo,UI_score=UI_score,ui_train=ui_train,top_K_list=top_K_list),eval_uies)
        pool.close()
        pool.join()
        for res_ in res:
            for top_k in top_K_list:
                HR_[f"{top_k}"].extend(res_[0][f"{top_k}"])
                NDCG_[f"{top_k}"].extend(res_[1][f"{top_k}"])
    else:
        return evaluate_one_ui_emo(eval_uies,UI_score,ui_train,top_K_list)

def metrics(eval_users, UI_score, eval_items, ui_train, top_K_list=[10]):
    HR = {}
    NDCG = {}
    for top_K in top_K_list:
        HR[f"{top_K}"] = []
        NDCG[f"{top_K}"] = []
    for i, u in enumerate(eval_users):
        target_item_index = eval_items[i]
        scores = UI_score[u]
        indexes = np.array([i for i in range(len(scores))])
        train_u_items = list(set(ui_train[u]))
        if target_item_index in train_u_items:
            train_u_items.remove(target_item_index)
        if train_u_items==[]:
            pass
        else:
            scores = np.delete(scores,train_u_items)
            indexes = np.delete(indexes,train_u_items)
        score_sort = sorted([(indexes[i], scores[i]) for i in range(len(scores))], key=lambda x: x[1], reverse=True)
        rec_list = [item[0] for item in score_sort]
        for top_K in top_K_list:
            rec_list_K = rec_list[:top_K]
            hr = hit_rate(target_item_index,rec_list_K)
            ndcg = ndcg_(target_item_index,rec_list_K)
            HR[f"{top_K}"].append(hr)
            NDCG[f"{top_K}"].append(ndcg)
    return HR, NDCG

def hit_rate(item, rec_list):
    if item in rec_list:
        return 1
    else:
        return 0

def ndcg_(item, rec_list):
    if item in rec_list:
        index_ = rec_list.index(item)+1
        return 1/np.log2(index_+1)
    else:
        return 0

def load_npy(path):
    return np.load(path)

class rating():
    def __init__(self,user_embedding,item_i_embedding,item_j_embedding,user_preferred_emo,item_i_emotion,item_j_emotion):
        self.user_embedding = user_embedding
        self.item_i_embedding = item_i_embedding
        self.item_j_embedding = item_j_embedding
        self.user_preferred_emo = user_preferred_emo
        self.item_i_emotion = item_i_emotion
        self.item_j_emotion = item_j_emotion

    def get_weights(self, weight_shape, name_=None):
        w = tf.compat.v1.get_variable(name=name_,
                                      shape=weight_shape,
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(mean=0, stddev=0.5))
        return w

    def get_bias(self, bias_shape, name_=None):
        b = tf.compat.v1.get_variable(name=name_,
                                      shape=bias_shape,
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(value=0))
        return b

    def bpr(self):
        u_rep = tf.compat.v1.concat((self.user_embedding,self.user_preferred_emo),axis=-1)
        i_rep = tf.compat.v1.concat((self.item_i_embedding,self.item_i_emotion),axis=-1)
        j_rep = tf.compat.v1.concat((self.item_j_embedding,self.item_j_emotion),axis=-1)
        # score
        u_i = tf.compat.v1.multiply(u_rep,i_rep,name='bpr_i_rating')
        u_j = tf.compat.v1.multiply(u_rep,j_rep,name='bpr_j_rating')
        uij = tf.compat.v1.reduce_sum(u_i-u_j,axis=1,keepdims=True)
        auc = tf.compat.v1.reduce_mean(tf.cast(uij>0,dtype=tf.float32))
        bpr_loss = -tf.compat.v1.reduce_mean(tf.compat.v1.math.log(tf.compat.v1.sigmoid(uij)))
        embedding_l2_loss = tf.compat.v1.add_n([tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.user_embedding,self.user_embedding)),
                                                tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.item_i_embedding,self.item_i_embedding)),
                                                tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.item_j_embedding,self.item_j_embedding))])
        return auc, bpr_loss, embedding_l2_loss

    def fnn(self,fnn_layers=[73,32]):
        u_rep = tf.compat.v1.concat((self.user_embedding, self.user_preferred_emo), axis=-1)
        i_rep = tf.compat.v1.concat((self.item_i_embedding, self.item_i_emotion), axis=-1)
        j_rep = tf.compat.v1.concat((self.item_j_embedding, self.item_j_emotion), axis=-1)
        # score
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        for i, h_size in enumerate(fnn_layers):
            if i == 0:
                fnnuw = self.get_weights(weight_shape=[73,h_size],name_=f'fnnu_w{i}')
                fnnub = self.get_weights(weight_shape=[h_size],name_=f'fnnu_b{i}')
                u_h = tf.compat.v1.add(tf.compat.v1.matmul(u_rep,fnnuw),fnnub)
                u_h = tf.compat.v1.nn.relu(u_h)#
                fnniw = self.get_weights(weight_shape=[73,h_size],name_=f'fnni_w{i}')
                fnnib = self.get_bias(bias_shape=[h_size], name_=f'fnni_b{i}')
                i_h = tf.compat.v1.add(tf.compat.v1.matmul(i_rep, fnniw), fnnib)
                i_h = tf.compat.v1.nn.relu(i_h)#
                j_h = tf.compat.v1.add(tf.compat.v1.matmul(j_rep, fnniw), fnnib)
                j_h = tf.compat.v1.nn.relu(j_h)#
                regularization = regularizer(fnnuw) + regularizer(fnniw)
            else:
                fnnuw = self.get_weights(weight_shape=[fnn_layers[i-1], h_size], name_=f'fnnu_w{i}')
                fnnub = self.get_weights(weight_shape=[h_size], name_=f'fnnu_b{i}')
                u_h = tf.compat.v1.add(tf.compat.v1.matmul(u_h, fnnuw), fnnub)
                u_h = tf.compat.v1.nn.relu(u_h)#
                fnniw = self.get_weights(weight_shape=[fnn_layers[i-1], h_size], name_=f'fnni_w{i}')
                fnnib = self.get_bias(bias_shape=[h_size], name_=f'fnni_b{i}')
                i_h = tf.compat.v1.add(tf.compat.v1.matmul(i_h, fnniw), fnnib)
                i_h = tf.compat.v1.nn.relu(i_h)#
                j_h = tf.compat.v1.add(tf.compat.v1.matmul(j_h, fnniw), fnnib)
                j_h = tf.compat.v1.nn.relu(j_h)#
                regularization = regularization + regularizer(fnnuw) + regularizer(fnniw)
        u_i = tf.compat.v1.multiply(u_h, i_h, name='bpr_i_rating')
        u_j = tf.compat.v1.multiply(u_h, j_h, name='bpr_j_rating')
        uij = tf.compat.v1.reduce_sum(u_i - u_j, axis=1, keepdims=True)
        auc = tf.compat.v1.reduce_mean(tf.cast(uij > 0, dtype=tf.float32))
        bpr_loss = -tf.compat.v1.reduce_mean(tf.compat.v1.math.log(tf.compat.v1.sigmoid(uij)))
        embedding_l2_loss = tf.compat.v1.add_n(
            [tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.user_embedding, self.user_embedding)),
             tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.item_i_embedding, self.item_i_embedding)),
             tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.item_j_embedding, self.item_j_embedding))]) + regularization
        return auc, bpr_loss, embedding_l2_loss, u_h, i_h

    def mlp(self,mlp_layers=[73*2,32,1]):
        u_rep = tf.compat.v1.concat((self.user_embedding, self.user_preferred_emo), axis=-1)
        i_rep = tf.compat.v1.concat((self.item_i_embedding, self.item_i_emotion), axis=-1)
        j_rep = tf.compat.v1.concat((self.item_j_embedding, self.item_j_emotion), axis=-1)
        # score
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        ui_concat = tf.compat.v1.concat((u_rep,i_rep),axis=-1)
        uj_concat = tf.compat.v1.concat((u_rep,j_rep),axis=-1)
        for i,h_size in enumerate(mlp_layers):
            if i==0:
                mlpw = self.get_weights(weight_shape=[73*2,h_size],name_=f'mlpw{i}')
                mlpb = self.get_bias(bias_shape=[h_size],name_=f'mlpb{i}')
                ui = tf.compat.v1.add(tf.compat.v1.matmul(ui_concat,mlpw),mlpb)
                ui = tf.compat.v1.nn.relu(ui)
                uj = tf.compat.v1.add(tf.compat.v1.matmul(uj_concat,mlpw),mlpb)
                uj = tf.compat.v1.nn.relu(uj)
                regularization = regularizer(mlpw)
            else:
                mlpw = self.get_weights(weight_shape=[mlp_layers[i-1], h_size], name_=f'mlpw{i}')
                mlpb = self.get_bias(bias_shape=[h_size], name_=f'mlpb{i}')
                ui = tf.compat.v1.add(tf.compat.v1.matmul(ui, mlpw), mlpb)
                ui = tf.compat.v1.nn.relu(ui)
                uj = tf.compat.v1.add(tf.compat.v1.matmul(uj, mlpw), mlpb)
                uj = tf.compat.v1.nn.relu(uj)
                regularization = regularization + regularizer(mlpw)
        uij = tf.compat.v1.reduce_sum(ui - uj, axis=1, keepdims=True)
        auc = tf.compat.v1.reduce_mean(tf.cast(uij > 0, dtype=tf.float32))
        bpr_loss = -tf.compat.v1.reduce_mean(tf.compat.v1.math.log(tf.compat.v1.sigmoid(uij)))
        embedding_l2_loss = tf.compat.v1.add_n(
            [tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.user_embedding, self.user_embedding)),
             tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.item_i_embedding, self.item_i_embedding)),
             tf.compat.v1.reduce_sum(
                 tf.compat.v1.multiply(self.item_j_embedding, self.item_j_embedding))]) + regularization
        return auc, bpr_loss, embedding_l2_loss, ui