import math
import numpy as np
from tqdm import tqdm

def eval_one_rank(pred_dict,label_index,K):
    if K>len(pred_dict):
        print('recommendation list length is less than K, you should set a correct K value')
    else:
        pred_list = sorted(pred_dict.items(), key=lambda x: x[1],reverse=True)[:K]
        indexes = [item[0] for item in pred_list]
        if label_index in indexes:
            hit = 1
            index_ = indexes.index(label_index)
            ndcg = math.log(2)/math.log(index_+2)
        else:
            hit = 0
            ndcg = 0
        return hit, ndcg

def bpr_rec_evaluate(rec_list,nega_nums,K):
    if K>len(rec_list):
        print('recommendation list length is less than K, you should set a correct K value')
    else:
        i_num = int(len(rec_list)/(nega_nums+1))
        hits = []
        ndcgs = []
        for i_ in tqdm(range(i_num)):
            u_rec = rec_list[i_*(nega_nums+1):(i_+1)*(nega_nums+1),]
            item_index = 0
            u_rec_sort = list(np.argsort(u_rec,axis=0)[::-1][:,0])[:K]# top-K
            if item_index in u_rec_sort:
                hits.append(1)
                index_ = u_rec_sort.index(item_index)
                ndcgs.append(math.log(2)/math.log(index_+2))
            else:
                hits.append(0)
                ndcgs.append(0)
        return hits, ndcgs

