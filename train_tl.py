import os
import glob
import time
import math
import numpy as np
import skvideo.io
import tensorflow as tf
import tensorlayer as tl
from model_tl import discriminator_I, discriminator_V,generator_I, getGRU

seed = 0
np.random.seed(seed)
batch_size = 16
n_iter = 120000

''' prepare dataset '''

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path+'/*')
videos = [ skvideo.io.vread(file) for file in files ]
# transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
videos = [ video.transpose(3, 0, 1, 2) / 255.0 for video in videos ]


''' prepare video sampling '''

n_videos = len(videos)
T = 16

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :]

# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    start = np.random.randint(0, noise.size(1) - (T+1))
    end = start + T
    return noise[:, start:end, :, :, :]

def random_choice():
    X = []
    print(videos.shape)
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        print('video shape:')
        print(video.shape)
        X.append(video)
    X = tf.stack(X)
    return X

# video length distribution
video_lengths = [video.shape[1] for video in videos]


''' set models '''

img_size = 96
nc = 3
ndf = 64 # from dcgan
ngf = 64
d_E = 10
hidden_size = 100 # guess
d_C = 50
d_M = d_E
nz  = d_C + d_M

lr = 0.0002
real_value = 0.9
dis_i = discriminator_I(nc, ndf)
dis_v = discriminator_V(nc, ndf,T)
gen_i = generator_I(nc, ngf, nz)
gru = getGRU(d_E, hidden_size)

''' prepare for train '''
label = tf.zeros([])

def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models')

def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)

''' setup optimizer '''

optim_di = tf.optimizers.Adam(lr,0.5,0.999)
optim_dv = tf.optimizers.Adam(lr,0.5,0.999)
optim_gi = tf.optimizers.Adam(lr,0.5,0.999)
optim_gru = tf.optimizers.Adam(lr,0.5,0.999)

criterion = tl.cost.binary_cross_entropy

''' generate input noise for fake vedio '''

def gen_z(n_frames):
    z_C = tf.Variable(tf.random.normal((batch_size,d_C)))
    z_C = tf.expand_dims(z_C,1)
    z_C = tf.tile(z_C,(1,n_frames,1))
    eps = tf.Variable(tf.random.normal((batch_size, d_E)))
    gru.initHidden(batch_size)
    z_M = tf.transpose(gru(eps, n_frames),[1,0])
    z = tf.concat([z_M,z_C],2)
    z = tf.reshape(z,(batch_size, n_frames, nz, 1, 1))
    return z

''' calculate gradient and back propagation '''
def bp(real_img, real_videos, n_frames):
    print("IN BP")
    print("real_img.shape=")
    print(real_img.shape)
    print("real_videos.shape=")
    print(real_videos.shape)
    print("n_frames.shape=")
    print(n_frames.shape)
    with tf.GradientTape(persistent=True) as tape:
        output_real_I = dis_i(real_img)
        output_real_V = dis_v(real_videos)
        ''''''
        print("output real")
        print(output_real_I)
        print(output_real_I.shape)
        print(output_real_V)
        print(output_real_V.shape)
        ''''''
        Z = gen_z(n_frames)
        Z = trim_noise(Z)
        #generate videos
        Z = tf.reshape(Z,(batch_size*T,nz,1,1))
        fake_videos = gen_i(Z)
        fake_videos = fake_videos.reshape((batch_size,T,nc,img_size,img_size))
        # transpose => (batch_size, nc, T, img_size, img_size)
        fake_videos = tf.transpose(fake_videos,[0,2,1,3,4])
        #sample image
        fake_img = fake_videos[:,:,np.random.randint(0,T),:,:]
        output_fake_I = dis_i(fake_img)
        output_fake_V = dis_v(fake_videos)
        ''''''
        print("output fake")
        print(output_fake_I)
        print(output_fake_I.shape)
        print(output_fake_V)
        print(output_fake_V.shape)
        ''''''
        err_real_I = criterion(output_real_I, 0.9)
        err_real_V = criterion(output_real_V, 0.9)
        err_fake_I = criterion(output_fake_I, 0.0)
        err_fake_V = criterion(output_fake_V, 0.0)
        err_I = err_real_I + err_fake_I
        err_V = err_real_V + err_fake_V

        err_fake_GI = criterion(output_fake_I, 0.9)
        err_fake_Gv = criterion(output_fake_V, 0.9)
    grad_di = tape.gradient(err_I, dis_i.trainable_weights)
    grad_dv = tape.gradient(err_V, dis_v.trainable_weights)
    grad_gi = tape.gradient(err_fake_GI, gen_i.trainable_weights)
    grad_gru = tape.gradient(err_fake_Gv, gru.trainable_weights)

    return err_I, err_V, err_fake_GI, err_fake_Gv


''' train models '''
start_time = time.time()
dis_i.train()
dis_v.train()
gen_i.train()
gru.train()
print('(%s) Begin training'%(timeSince(start_time)))
for epoch in range(1, n_iter+1):
    ''' prepare real images '''
    real_videos = random_choice()
    real_videos = tf.Variable(real_videos)
    real_img = real_videos[:,:,np.random.randint(0,T),:,:]

    ''' prepare fake images '''
    n_frames = video_lengths[np.random.randint(0,n_videos)]
    err_I, err_V, err_fake_Gi, err_fake_Gv = bp(real_img, real_videos, n_frames)

    if epoch % 100 == 0:
        print('[%d/%d] (%s) Err_I: %.4f Err_V: %.4f Err_fake_Gi: %.4f Err_fake_Gv: %.4f'%(epoch,n_iter,timeSince(start_time),err_I,err_V,err_fake_Gi,err_fake_Gv))

dis_i.save('discriminator_I.h5')
dis_v.save('discriminator_V.h5')
gen_i.save('generator_I.h5')
gru.save('GRU.h5')
