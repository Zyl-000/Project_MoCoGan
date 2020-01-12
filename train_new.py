import os
import glob
import time
import math
import numpy as np
import skvideo.io
import tensorflow as tf
import tensorlayer as tl
from new import discriminator_I, discriminator_V, generator_I, getGRU

seed = 0
np.random.seed(seed)
batch_size = 16
n_iter = 1000

''' prepare dataset '''

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path + '/*')
videos = [skvideo.io.vread(file) for file in files]
# transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
videos = [video.transpose(3, 0, 1, 2) / 255.0 for video in videos]

''' prepare video sampling '''

n_videos = len(videos)
T = 16


# for true video


def trim(video):
    start = np.random.randint(0, video.shape[1] - (T + 1))
    end = start + T
    return video[:, start:end, :, :]


# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency


def trim_noise(noise):
    start = np.random.randint(0, noise.size(1) - (T + 1))
    end = start + T
    return noise[:, start:end, :, :, :]


def random_choice():
    X = []
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos - 1)]
        video = tf.convert_to_tensor(trim(video))
        X.append(video)
    X = tf.stack(X)
    return X


# video length distribution
video_lengths = [video.shape[1] for video in videos]

''' set models '''

img_size = 96
nc = 3
ndf = 64
ngf = 64
d_E = 10
hidden_size = T
d_C = 50
d_M = d_E
nz = d_C + d_M

lr = 0.0002
dis_i = discriminator_I(nc, ndf)
dis_v = discriminator_V(nc, ndf, T)
gen_i = generator_I((batch_size * T, 1, 1, nz), nc, ngf, nz)
gru = getGRU(d_E, hidden_size)

''' prepare for train '''
label = tf.zeros([])


def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60 ** 2) * 24))
    h = math.floor(s / (60 ** 2)) - d * 24
    m = math.floor(s / 60) - h * 60 - d * 24 * 60
    s = s - m * 60 - h * (60 ** 2) - d * 24 * (60 ** 2)
    return '%dd %dh %dm %ds' % (d, h, m, s)


trained_path = os.path.join(current_path, 'trained_models')


def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)


''' setup optimizer '''

optim_di = tf.optimizers.Adam(lr, 0.5, 0.999)
optim_dv = tf.optimizers.Adam(lr, 0.5, 0.999)
optim_gi = tf.optimizers.Adam(lr, 0.5, 0.999)
optim_gru = tf.optimizers.Adam(lr, 0.5, 0.999)

criterion = tf.losses.binary_crossentropy

''' generate input noise for fake vedio '''

''' calculate gradient and back propagation '''


def bp(real_img_, real_videos_):
    with tf.GradientTape(persistent=True) as tape:
        real_img_ = tf.cast(real_img_, 'float32')
        real_videos_ = tf.cast(real_videos, 'float32')
        real_img_ = tf.reshape(real_img_, (batch_size, 96, 96, nc))
        real_videos_ = tf.reshape(real_videos_, (batch_size, T, 96, 96, 3))
        output_real_I = tf.reshape(dis_i(real_img_),(batch_size,1))
        output_real_V = tf.reshape(dis_v(real_videos_),(batch_size,1))

        z_C = tf.random.normal((batch_size, d_C))
        z_C = tf.expand_dims(z_C, 1)
        z_C = tf.tile(z_C, (1, T, 1))
        eps = tf.random.normal((batch_size, hidden_size, d_E), dtype=tf.float32)
        z_M = gru(inputs=eps)

        z = tf.concat([z_M, z_C], 2)
        # print(z.shape)
        Z = tf.reshape(z, (batch_size * T, 1, 1, nz))
        # generate videos
        Z = tf.cast(Z, dtype=tf.float32)
        # print(Z.shape)
        fake_videos = gen_i(Z)
        '''
        print('after gene')
        print(fake_videos.shape)
        '''
        fake_videos = tf.reshape(fake_videos, (batch_size, T, img_size, img_size, nc))
        '''
        print("##################################################################")
        print(fake_videos.shape)
        '''
        # sample image
        fake_img = fake_videos[:, np.random.randint(0, T), :, :, :]
        # print(fake_img.shape)
        output_fake_I = tf.reshape(dis_i(fake_img),(batch_size,1))
        output_fake_V = tf.reshape(dis_v(fake_videos),(batch_size,1))
        '''
        print('output real I:')
        print(output_real_I)
        print('outout real V:')
        print(output_real_V)
        print('output fake I:')
        print(output_fake_I)
        print('output fake V:')
        print(output_fake_V)
        '''
        err_real_I = criterion(output_real_I, tl.alphas_like(output_real_I, 1.0))
        err_real_V = criterion(output_real_V, tl.alphas_like(output_real_V, 1.0))
        err_fake_I = criterion(output_fake_I, tl.alphas_like(output_fake_I, 0.0))
        err_fake_V = criterion(output_fake_V, tl.alphas_like(output_fake_V, 0.0))
        err_I = err_real_I + err_fake_I
        err_V = err_real_V + err_fake_V
        err_fake_Gi = criterion(output_fake_I, tl.alphas_like(output_fake_I, 1.0))
        err_fake_Gv = criterion(output_fake_V, tl.alphas_like(output_fake_V, 1.0))

    grad_gi = tape.gradient(err_fake_Gi, gen_i.trainable_weights)
    optim_gi.apply_gradients(zip(grad_gi, gen_i.trainable_weights))
    grad_gru = tape.gradient(err_fake_Gv, gru.trainable_weights)
    optim_gru.apply_gradients(zip(grad_gru, gru.trainable_weights))
    '''
    print('the generator:')
    print(gen_i.trainable_weights)
    print(gru.trainable_weights)
    '''
    grad_di = tape.gradient(err_I, dis_i.trainable_weights)
    optim_di.apply_gradients(zip(grad_di, dis_i.trainable_weights))
    grad_dv = tape.gradient(err_V, dis_v.trainable_weights)
    optim_dv.apply_gradients(zip(grad_dv, dis_v.trainable_weights))

    #print(grad_gi)
    #print(grad_gru)
    del tape
    '''
    print('FINALLY BP OVER')
    print(err_I)
    print(err_V)
    print(err_fake_Gi)
    print(err_fake_Gv)
    '''
    return err_I, err_V, err_fake_Gi, err_fake_Gv


''' train models '''
start_time = time.time()
dis_i.train()
dis_v.train()
gen_i.train()
gru.train()
print('(%s) Begin training' % (timeSince(start_time)))
for epoch in range(1, n_iter+1):
    ''' prepare real images '''
    real_videos = random_choice()
    # real_videos = tf.Variable(real_videos)
    real_img = real_videos[:, :, np.random.randint(0, T), :, :]

    ''' back propagation '''
    err_I, err_V, err_fake_Gi, err_fake_Gv = bp(real_img, real_videos)

    if epoch % 10 == 0:
        print('########[%d/%d] (%s) Err_I: %.4f Err_V: %.4f Err_fake_Gi: %.4f Err_fake_Gv: %.4f'
              % (epoch, n_iter, timeSince(start_time), tf.reduce_mean(err_I), tf.reduce_mean(err_V),
                 tf.reduce_mean(err_fake_Gi), tf.reduce_mean(err_fake_Gv)))

dis_i.save_weights('discriminator_I.h5')
dis_v.save_weights('discriminator_V.h5')
gen_i.save_weights('generator_I.h5')
gru.save_weights('GRU.h5')
