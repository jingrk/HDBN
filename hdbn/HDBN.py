import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import time
import math

from model_modules.vae import VAE, Embedding
from utils.configs import Paths # Configs
from utils.data_utils import data_RW, data_prepare
from utils.evaluation import rating, evaluate_all_emo

hyper_param = {
    'lr': 0.05,
    'emb_size': 64,
    'neg_num': 10,
    'batch_size': 512,
    'epochs': 100,
    'seed': 42,
    'top_K': [5,10,15,20],
    'emo_num': 132,
    'emo_emb_size': 16,
    'vae1_enc': [64,64],
    'vae1_dec': [64,64],
    'vae1_z': 16,
    'vae2_enc': [64,64],
    'vae2_dec': [64,64],
    'vae2_z': 16,
    'mu_prior': 0,
    'sigma_prior': 1,
    'KL_gamma1': 0.01,
    'KL_gamma2': 0.05,
    'recon_beta1': 0.000001,
    'recon_beta2': 0.0001
}



groups_data = {}
for group_ in range(50):
    group_set = data_RW().dataframe_load(f"./data/EmoMusicLJ/cluster_users_train/users_C{group_}.csv")
    content = data_prepare().get_sets(group_set, emo=True)
    groups_data[group_] = content

'''--------------------------------------------------------------------------------------------'''
u2group = np.load(Paths['U2Cluster'])#shape=(12557,1)
songs_emo = np.load(Paths['songs_emo_audio']).astype(np.float32)

train_set = data_RW().dataframe_load(Paths['dirpath']+Paths['UI_train'])
ui_train, items_train, unum_train, inum_train, ue_train = data_prepare().get_sets(train_set,emo=True)
valid_uies_group = {}
for group_ in range(50):
    valid_set_group = data_RW().dataframe_load(f"./data/EmoMusicLJ/cluster_users_valid/users_C{group_}.csv")
    valid_uies = []
    for i in range(len(valid_set_group)):
        valid_uies.append([valid_set_group['user_id'][i],valid_set_group['song_id'][i],valid_set_group['emo_id'][i]])
    valid_uies = np.array(valid_uies)
    valid_uies_group[group_]=valid_uies

all_iids = np.array([i for i in range(6095)])

print(f"train users #{unum_train} items #{inum_train}")
'''--------------------------------------------------------------------------------------------'''

## load the trained emotion selection models
def reparameteriztion(mu,rho):
    epsilon = tf.compat.v1.random.normal(shape=tf.shape(rho),mean=0.0,stddev=1.0)# random seed
    sigma = tf.compat.v1.log1p(tf.compat.v1.exp(rho))
    reparams = mu + epsilon*sigma
    return reparams

def get_trained_params(path,model_name):
    trained_params = {}
    saver = tf.compat.v1.train.import_meta_graph(path+model_name+'.meta')#,clear_devices=True
    with tf.compat.v1.Session() as sess:
        saver.restore(sess,path+model_name)
        for var in tf.compat.v1.trainable_variables():
            trained_params[var.name] = sess.run(var)
    return trained_params

pre_Models = {}
for cluster_n in range(50):
    PATH = "./pretrained_psi/"
    model_name = f"emo_select_model_g{cluster_n}"
    if cluster_n<10:
        model_path = PATH+f"0{cluster_n}/"
    else:
        model_path = PATH+f"{cluster_n}/"
    trained_PARAMS = get_trained_params(path=model_path,model_name=model_name)
    pre_Models[cluster_n] = trained_PARAMS
    tf.compat.v1.reset_default_graph()

def pretrained_model(user_emo,cluster_n):
    params = pre_Models[cluster_n]
    w1mu = params[f"c{cluster_n}_w_mu_1:0"]
    w1rho = params[f"c{cluster_n}_w_rho_1:0"]
    w1 = reparameteriztion(w1mu,w1rho)
    b1mu = params[f"c{cluster_n}_bia_mu_1:0"]
    b1rho = params[f"c{cluster_n}_bia_rho_1:0"]
    b1 = reparameteriztion(b1mu,b1rho)
    h1 = tf.compat.v1.add(tf.compat.v1.matmul(user_emo,w1),b1)
    h1 = tf.compat.v1.nn.relu(h1)
    w2mu = params[f"c{cluster_n}_w_mu_2:0"]
    w2rho = params[f"c{cluster_n}_w_rho_2:0"]
    w2 = reparameteriztion(w2mu,w2rho)
    b2mu =  params[f"c{cluster_n}_bia_mu_2:0"]
    b2rho = params[f"c{cluster_n}_bia_rho_2:0"]
    b2 = reparameteriztion(b2mu,b2rho)
    h2 = tf.compat.v1.add(tf.compat.v1.matmul(h1,w2),b2)
    h2 = tf.compat.v1.nn.relu(h2)
    wout = params[f"c{cluster_n}_wout:0"]
    bout = params[f"c{cluster_n}_bout:0"]
    emo_select = tf.compat.v1.add(tf.compat.v1.matmul(h2,wout),bout)
    emo_select = tf.compat.v1.nn.softmax(emo_select,axis=-1)
    return emo_select

def emo_select_compute(user_emo, cluster_n):
    cases = [(tf.compat.v1.equal(cluster_n,n), lambda n=n: pretrained_model(user_emo, n)) for n in range(50)]
    return tf.compat.v1.case(cases,default=lambda: np.array([1,1,1,1,1,1,1,1,1],dtype=np.float32),exclusive=False)


def placeholder(data_type, shape_, name_):
    return tf.compat.v1.placeholder(dtype=data_type,shape=shape_,name=name_)


u_p = placeholder(data_type=tf.compat.v1.int32,shape_=[None],name_='user_index')
i_p = placeholder(data_type=tf.compat.v1.int32,shape_=[None],name_='item_index')
j_p = placeholder(data_type=tf.compat.v1.int32,shape_=[None],name_='neg_item_index')
e_p = placeholder(data_type=tf.compat.v1.int32,shape_=[None],name_='emo_index')
g_p = placeholder(data_type=tf.compat.v1.int32,shape_=[None],name_='user_group')

u_emb = Embedding().user_emb(user_num=unum_train,emb_size=hyper_param["emb_size"])
i_emb = Embedding().item_emb(item_num=inum_train,emb_size=hyper_param["emb_size"])
e_emb = Embedding().emo_emb(emo_num=hyper_param["emo_num"],emb_size=hyper_param["emo_emb_size"])


u_p_emb = tf.compat.v1.nn.embedding_lookup(u_emb,u_p)
i_p_emb = tf.compat.v1.nn.embedding_lookup(i_emb,i_p)
j_p_emb = tf.compat.v1.nn.embedding_lookup(i_emb,j_p)
e_p_emb = tf.compat.v1.nn.embedding_lookup(e_emb,e_p)
i_emo = tf.compat.v1.gather(songs_emo,i_p)
j_emo = tf.compat.v1.gather(songs_emo,j_p)

u_p_emb_copy = tf.compat.v1.stop_gradient(tf.compat.v1.gather(u_emb,u_p))
vae1_mu, vae1_rho = VAE().encoder(input_x=u_p_emb_copy,z_dim=hyper_param['vae1_z'],units=hyper_param['vae1_enc'],vae_num='vae1')

vae1_z, vae1_sigma = VAE().reparam(vae1_mu,vae1_rho)
vae1_recon = VAE().decoder(input_z=vae1_z,recon_dim=hyper_param['emb_size'],units=hyper_param['vae1_dec'],vae_num='vae1')

KL_vae1 = tf.compat.v1.reduce_mean(
    tf.compat.v1.reduce_sum(0.5*(
            -tf.compat.v1.log(tf.compat.v1.square(vae1_sigma)+1e-7)-1+tf.compat.v1.square(vae1_sigma)+tf.compat.v1.square(vae1_mu)
    ),axis=-1))

recon_loss_vae1 = tf.compat.v1.reduce_mean(
    tf.compat.v1.reduce_sum(
        tf.compat.v1.pow((u_p_emb_copy-vae1_recon),2)
        ,axis=-1))

e_p_emb_copy = tf.compat.v1.stop_gradient(tf.compat.v1.gather(e_emb,e_p))
vae2_mu, vae2_rho = VAE().encoder(input_x=e_p_emb_copy,z_dim=hyper_param['vae2_z'],units=hyper_param['vae2_enc'],vae_num='vae2')
vae2_z, vae2_sigma = VAE().reparam(vae2_mu,vae2_rho)
vae2_recon = VAE().decoder(input_z=vae2_z,recon_dim=hyper_param['emo_emb_size'],units=hyper_param['vae2_dec'],vae_num='vae2')

KL_vae2 = tf.compat.v1.reduce_mean(
    tf.compat.v1.reduce_sum(
        0.5*(tf.compat.v1.log(1e-7+tf.compat.v1.square(vae1_sigma)/(tf.compat.v1.square(vae2_sigma)+1e-7))-1+
             tf.compat.v1.square(vae2_sigma/(vae1_sigma+1e-7))+
             tf.compat.v1.square((vae2_mu-vae1_mu)/(vae1_sigma+1e-7))
             ),axis=-1))

recon_loss_vae2 = tf.compat.v1.reduce_mean(
    tf.compat.v1.reduce_sum(
        tf.compat.v1.pow((e_p_emb_copy-vae2_recon),2)
    ,axis=-1)
)

u_p_clusters = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32,shape=(),name="user_group_number")
u_emo_select = emo_select_compute(vae2_z,u_p_clusters)
u_emo_select = tf.compat.v1.squeeze(u_emo_select)

auc, bpr_loss, emb_l2_loss = rating(u_p_emb,i_p_emb,j_p_emb,u_emo_select,i_emo,j_emo).bpr()
eval_score = tf.compat.v1.matmul(tf.compat.v1.concat((u_p_emb,u_emo_select),axis=-1),tf.compat.v1.transpose(tf.compat.v1.concat((i_p_emb,i_emo),axis=-1)))####


loss = bpr_loss + hyper_param['KL_gamma1']*KL_vae1\
       + hyper_param['KL_gamma2']*KL_vae2\
       + hyper_param['recon_beta1']*recon_loss_vae1\
       + hyper_param['recon_beta2']*recon_loss_vae2\
       + hyper_param['l2_lambda']*emb_l2_loss

global_step = tf.compat.v1.Variable(tf.compat.v1.constant(0),trainable=False)
lr = tf.compat.v1.train.exponential_decay(learning_rate=hyper_param['lr'],
                                          global_step=global_step,
                                          decay_steps=1024,
                                          decay_rate=0.99,
                                          staircase=True,
                                          name='lr_decay')

Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
gradients, variables = zip(*Optimizer.compute_gradients(loss))

clipped_gradients, _ = tf.compat.v1.clip_by_global_norm(gradients,clip_norm=10)
Opt = Optimizer.apply_gradients(zip(clipped_gradients,variables))

print(hyper_param)
saver = tf.compat.v1.train.Saver(max_to_keep=10)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(f"-----------training------------")
    T0 = time.time()
    for epoch in range(hyper_param['epochs']):
        GROUP = shuffle([g_ for g_ in range(50)])
        # negative sampling of all training groups in each epoch
        GROUP_data = {}
        # generate the negative sampling train data
        for group_ in GROUP:
            ui_train_group, items_train_group, unum_train_group, inum_train_group, ue_train_group = groups_data[group_]
            train_data_ = shuffle(data_prepare().negative_sampling_emo(ui_train_group, items_train, ue_train_group,hyper_param['neg_num']))
            GROUP_data[group_] = train_data_
        if epoch==0:# save the metrics from the random initialized weights
            L0 = []
            Lbpr_ = []
            AUC_0 = []
            KL1_0 = []
            KL2_0 = []
            Recon1_0 = []
            Recon2_0 = []
            L2_L = []
            t0 = time.time()
            for group_ in GROUP:
                train_data = GROUP_data[group_]
                batch_num = int(len(train_data) / hyper_param['batch_size'])
                for batch in range(batch_num):
                    if batch == (batch_num - 1):
                        x_input = train_data[batch * hyper_param['batch_size']:, :]
                    else:
                        x_input = train_data[batch * hyper_param['batch_size']:(batch + 1) * hyper_param['batch_size'], :]
                    loss_, bpr_loss_, auc_, kl1_, kl2_, recon1_, recon2_, l2_ = sess.run([loss, bpr_loss, auc, KL_vae1, KL_vae2, recon_loss_vae1, recon_loss_vae2, emb_l2_loss],
                                                                                    feed_dict={u_p: x_input[:, 0],
                                                                                               i_p: x_input[:, 1],
                                                                                               j_p: x_input[:, 2],
                                                                                               e_p: x_input[:, 3],
                                                                                               u_p_clusters: group_})
                    L0.append(loss_)
                    Lbpr_.append(bpr_loss_)
                    AUC_0.append(auc_)
                    KL1_0.append(kl1_)
                    KL2_0.append(kl2_)
                    Recon1_0.append(recon1_)
                    Recon2_0.append(recon2_)
                    L2_L.append(l2_)
            t_score = 0
            t_hr_ndcg = 0
            HR_eval = {}
            NDCG_eval = {}
            for top_k_ in hyper_param['top_K']:
                HR_eval[f"{top_k_}"] = []
                NDCG_eval[f"{top_k_}"] = []
            for group_ in range(50):
                t_temp1 = time.time()
                valid_uies = valid_uies_group[group_]
                SCORE = sess.run(eval_score,feed_dict={u_p:valid_uies[:,0], i_p:all_iids, e_p:valid_uies[:,-1], u_p_clusters: group_})
                t_temp2 = time.time()
                t_score = t_score + t_temp2-t_temp1
                HR, NDCG = evaluate_all_emo(valid_uies, SCORE, ui_train, hyper_param['top_K'], process_num=1)
                for top_k_ in hyper_param['top_K']:
                    HR_eval[f"{top_k_}"].extend(HR[f"{top_k_}"])
                    NDCG_eval[f"{top_k_}"].extend(NDCG[f"{top_k_}"])
                t_hr_ndcg = t_hr_ndcg + t_temp3 - t_temp2
            t3 = time.time()
            print("Epoch Train_loss BPR_loss Train_AUC Train_KL1 Train_KL2 Train_recon1 Train_recon2 Train_L2 "
                  "HR@5 HR@10 HR@15 HR@20 NDCG@5 NDCG@10 NDCG@15 NDCG@20")
            print(f"{epoch} {np.mean(L0)} {np.mean(Lbpr_)} {np.mean(AUC_0)} {np.mean(KL1_0)} {np.mean(KL2_0)} {np.mean(Recon1_0)} {np.mean(Recon2_0)} {np.mean(L2_L)} "
                  f"{[np.mean(HR_eval[str(K)]) for K in hyper_param['top_K']]} {[np.mean(NDCG_eval[str(K)]) for K in hyper_param['top_K']]}")

        for group_ in GROUP:
            train_data = GROUP_data[group_]
            batch_num = int(len(train_data) / hyper_param['batch_size'])
            if batch_num * hyper_param['batch_size'] < len(train_data):
                batch_num = batch_num + 1
            for batch in range(batch_num):
                if batch == (batch_num - 1):
                    x_input = train_data[batch * hyper_param['batch_size']:, :]
                else:
                    x_input = train_data[batch * hyper_param['batch_size']:(batch + 1) * hyper_param['batch_size'], :]
                _ = sess.run([Opt], feed_dict={u_p: x_input[:, 0],
                                               i_p: x_input[:, 1],
                                               j_p: x_input[:, 2],
                                               e_p: x_input[:, 3],
                                               u_p_clusters: group_})
        Lr_ = sess.run(lr)
        L = []
        Lbpr = []
        AUC_ = []
        KL1_ = []
        KL2_ = []
        Recon1_ = []
        Recon2_ = []
        L2_L = []
        for group_ in GROUP:
            train_data = GROUP_data[group_]
            batch_num = int(len(train_data) / hyper_param['batch_size'])
            if batch_num * hyper_param['batch_size'] < len(train_data):
                batch_num = batch_num + 1
            for batch in range(batch_num):
                if batch == (batch_num - 1):
                    x_input = train_data[batch * hyper_param['batch_size']:, :]
                else:
                    x_input = train_data[batch * hyper_param['batch_size']:(batch + 1) * hyper_param['batch_size'], :]
                loss_, bpr_loss_, auc_, kl1_, kl2_, recon1_, recon2_, l2_ = sess.run([loss, bpr_loss, auc, KL_vae1, KL_vae2, recon_loss_vae1, recon_loss_vae2, emb_l2_loss],
                                                                                feed_dict={u_p: x_input[:, 0],
                                                                                           i_p: x_input[:, 1],
                                                                                           j_p: x_input[:, 2],
                                                                                           e_p: x_input[:, 3],
                                                                                           u_p_clusters: group_})
                L.append(loss_)
                Lbpr.append(bpr_loss_)
                AUC_.append(auc_)
                KL1_.append(kl1_)
                KL2_.append(kl2_)
                Recon1_.append(recon1_)
                Recon2_.append(recon2_)
                L2_L.append(l2_)

        t_score = 0
        t_hr_ndcg = 0
        HR_eval = {}
        NDCG_eval = {}
        for top_k_ in hyper_param['top_K']:
            HR_eval[f"{top_k_}"] = []
            NDCG_eval[f"{top_k_}"] = []
        for group_ in range(50):
            t_temp1 = time.time()
            valid_uies = valid_uies_group[group_]
            SCORE = sess.run(eval_score, feed_dict={u_p: valid_uies[:, 0], i_p: all_iids, e_p: valid_uies[:, -1],
                                                    u_p_clusters: group_})
            t_temp2 = time.time()
            t_score = t_score + t_temp2 - t_temp1
            HR, NDCG = evaluate_all_emo(valid_uies, SCORE, ui_train, hyper_param['top_K'], process_num=1)
            for top_k_ in hyper_param['top_K']:
                HR_eval[f"{top_k_}"].extend(HR[f"{top_k_}"])
                NDCG_eval[f"{top_k_}"].extend(NDCG[f"{top_k_}"])
            t_temp3 = time.time()
            t_hr_ndcg = t_hr_ndcg + t_temp3 - t_temp2
        print(f"{epoch+1} {np.mean(L)} {np.mean(Lbpr)} {np.mean(AUC_)} {np.mean(KL1_)} {np.mean(KL2_)} {np.mean(Recon1_)} {np.mean(Recon2_)} {np.mean(L2_L)} "
              f"{[np.mean(HR_eval[str(K)]) for K in hyper_param['top_K']]} {[np.mean(NDCG_eval[str(K)]) for K in hyper_param['top_K']]}")




