import tensorflow as tf

score_config = {
    'weights':[[100,32]],
    'keep_prob':0.75,
}

def paire_ranking(u_latent,i_latent,j_latent,train_mode,consider_content_preference=True):
    l2_norm = tf.constant(value=0,dtype=tf.float32)
    if consider_content_preference==True:
        u_i = tf.matmul(tf.transpose(u_latent,perm=[0,2,1]),tf.reshape(i_latent,shape=[i_latent.shape[0],1,50]))
        u_j = tf.matmul(tf.transpose(u_latent,perm=[0,2,1]),tf.reshape(j_latent,shape=[j_latent.shape[0],1,50]))
        print('u_i shape',u_i.shape,'u_j shape',u_j.shape)
        w = tf.get_variable(name='ui_w',shape=[u_i.shape[-1],u_i.shape[-1]],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))
        print('w shape',w.shape)
        l2_norm += tf.nn.l2_loss(w)
        u_i_w = tf.matmul(u_i,w)
        u_j_w = tf.matmul(u_j,w)
        print('u_i_w shape',u_i_w.shape,'u_j_shape',u_j_w.shape)
        ui_att = tf.nn.softmax(tf.matmul(u_latent,u_i_w))
        uj_att = tf.nn.softmax(tf.matmul(u_latent,u_j_w))
        print('ui_att shape',ui_att.shape,'uj_att shape',uj_att.shape)
        iu = tf.multiply(ui_att,u_latent)
        ju = tf.multiply(uj_att,u_latent)
        print('iu shape',iu.shape,'ju shape',ju.shape)
        u_i_conc = tf.concat([iu,tf.reshape(i_latent,shape=[i_latent.shape[0],1,50])],axis=2)
        u_j_conc = tf.concat([ju,tf.reshape(j_latent,shape=[j_latent.shape[0],1,50])],axis=2)
    else:
        u_i_conc = tf.concat([u_latent,tf.reshape(i_latent,shape=[i_latent.shape[0],1,50])],axis=2)
        u_j_conc = tf.concat([u_latent,tf.reshape(j_latent,shape=[j_latent.shape[0],1,50])],axis=2)
    print('u_i_conc shape',u_i_conc.shape,'u_j_conc shape',u_j_conc.shape)
    for i, shape in enumerate(score_config['weights']):
        w = tf.get_variable(name='w_score'+str(i),shape=shape,dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))
        b = tf.get_variable(name='b_score'+str(i),shape=shape[-1],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))
        u_i_conc = tf.nn.relu(tf.add(tf.matmul(u_i_conc,w),b))
        u_i_conc = tf.layers.dropout(inputs=u_i_conc,rate=1-score_config['keep_prob'],training=train_mode)
        u_j_conc = tf.nn.relu(tf.add(tf.matmul(u_j_conc,w),b))
        u_j_conc = tf.layers.dropout(inputs=u_j_conc,rate=1-score_config['keep_prob'],training=train_mode)
        l2_norm += tf.nn.l2_loss(w)
    w_score = tf.get_variable(name='output_w', shape=[score_config['weights'][-1][-1], 1], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    b_score = tf.get_variable(name='output_b', shape=[1], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    ui_score = tf.add(tf.matmul(u_i_conc,w_score),b_score)
    uj_score = tf.add(tf.matmul(u_j_conc,w_score),b_score)
    u_ij = tf.reduce_sum(ui_score-uj_score,axis=1,keep_dims=True)
    auc = tf.reduce_mean(tf.to_float(u_ij > 0))
    loss = -tf.reduce_mean(tf.log(tf.sigmoid(u_ij)))
    return ui_score, auc, loss, l2_norm