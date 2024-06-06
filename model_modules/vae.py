import tensorflow as tf

class VAE():
    def __init__(self):
        pass

    def get_weights(self,weight_shape,name_=None):
        w_init = tf.random_normal_initializer(mean=0, stddev=0.5)
        w = tf.compat.v1.get_variable(name=name_,
                                      shape=weight_shape,
                                      dtype=tf.float32,
                                      initializer=w_init)
        return w

    def get_bias(self,bias_shape,name_=None):
        b_init = tf.constant_initializer(value=0)
        b = tf.compat.v1.get_variable(name=name_,
                                      shape=bias_shape,
                                      dtype=tf.float32,
                                      initializer=b_init)
        return b

    def encoder(self,input_x,z_dim,units=None,vae_num=None):
        for i, unit_num in enumerate(units):
            if i==0:
                shape=[input_x.shape[-1],unit_num]
                w = self.get_weights(weight_shape=shape, name_=vae_num+f"encoder_w_{i}")
                b = self.get_bias(bias_shape=[unit_num], name_=vae_num+f"encoder_b_{i}")
                h = tf.compat.v1.add(tf.compat.v1.matmul(input_x,w),b)
                h = tf.compat.v1.nn.relu(h)
            else:
                shape=[units[i-1],unit_num]
                w = self.get_weights(weight_shape=shape, name_=vae_num+f"encoder_w_{i}")
                b = self.get_bias(bias_shape=[shape[-1]], name_=vae_num+f"encoder_b_{i}")
                h = tf.compat.v1.add(tf.compat.v1.matmul(h, w), b)
                h = tf.compat.v1.nn.relu(h)
        mu_w = self.get_weights(weight_shape=[h.shape[-1],z_dim],name_=vae_num+"z_mu_w")
        mu_b = self.get_bias(bias_shape=[z_dim],name_=vae_num+"z_mu_b")
        mu = tf.compat.v1.add(tf.compat.v1.matmul(h,mu_w),mu_b)
        rho_w = self.get_weights(weight_shape=[h.shape[-1],z_dim],name_=vae_num+"z_rho_w")
        rho_b = self.get_bias(bias_shape=[z_dim],name_=vae_num+"z_rho_b")
        rho = tf.compat.v1.add(tf.compat.v1.matmul(h,rho_w),rho_b)
        return mu, rho

    def reparam(self,mu, rho):
        epsilon = tf.compat.v1.random.normal(shape=tf.shape(rho),mean=0,stddev=1.0)
        sigma = tf.compat.v1.log1p(tf.compat.v1.exp(rho))
        z = mu + epsilon*sigma
        return z, sigma

    def decoder(self,input_z, recon_dim, units=None,vae_num=None):
        for i, unit_num in enumerate(units):
            if i == 0:
                shape = [input_z.shape[-1],units[i]]
                w = self.get_weights(weight_shape=shape,name_=vae_num+f"decoder_w_{i}")
                b = self.get_bias(bias_shape=[unit_num],name_=vae_num+f"decoder_b_{i}")
                h = tf.compat.v1.add(tf.compat.v1.matmul(input_z,w),b)
                h = tf.compat.v1.nn.relu(h)
            else:
                shape=[units[i-1],unit_num]
                w = self.get_weights(weight_shape=shape,name_=vae_num+f"decoder_w_{i}")
                b = self.get_bias(bias_shape=[unit_num],name_=vae_num+f"decoder_b_{i}")
                h = tf.compat.v1.add(tf.compat.v1.matmul(h,w),b)
                h = tf.compat.v1.nn.relu(h)
        recon_w = self.get_weights(weight_shape=[units[-1],recon_dim],name_=vae_num+"recon_w")
        recon_b = self.get_bias(bias_shape=[recon_dim],name_=vae_num+'recon_b')
        recon_x = tf.compat.v1.add(tf.compat.v1.matmul(h,recon_w),recon_b)
        return recon_x

class Embedding():
    def __init__(self):
        pass

    def user_emb(self,user_num, emb_size):
        u_emb = (tf.compat.v1.get_variable(name='user_embedding', shape=[user_num + 1, emb_size],
                                           initializer=tf.random_normal_initializer(mean=0, stddev=0.5),dtype=tf.float32)) / emb_size
        return u_emb

    def item_emb(self,item_num, emb_size):
        i_emb = (tf.compat.v1.get_variable(name='item_embedding', shape=[item_num + 1, emb_size],
                                           initializer=tf.random_normal_initializer(mean=0, stddev=0.5),dtype=tf.float32)) / emb_size
        return i_emb

    def emo_emb(self,emo_num, emb_size):
        e_emb = (tf.compat.v1.get_variable(name='emo_embedding', shape=[emo_num + 1, emb_size],
                                           initializer=tf.random_normal_initializer(mean=0, stddev=0.5),dtype=tf.float32)) / emb_size
        return e_emb

    def artists_emb(self,art_num, emb_size):
        art_emb = (tf.compat.v1.get_variable(name='artist_embedding', shape=[art_num + 1, emb_size],
                                             initializer=tf.random_normal_initializer(mean=0, stddev=0.5),
                                             dtype=tf.float32)) / emb_size
        return art_emb

    def years_emb(self,year_num, emb_size):
        year_emb = (tf.compat.v1.get_variable(name='year_embedding', shape=[year_num + 1, emb_size],
                                              initializer=tf.random_normal_initializer(mean=0, stddev=0.5),
                                              dtype=tf.float32)) / emb_size
        return year_emb

    def genres_emb(self,genre_num, emb_size):
        genre_emb = (tf.compat.v1.get_variable(name='genre_embedding', shape=[genre_num + 1, emb_size],
                                               initializer=tf.random_normal_initializer(mean=0, stddev=0.5),
                                               dtype=tf.float32)) / emb_size
        return genre_emb



