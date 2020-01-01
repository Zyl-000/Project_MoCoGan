import os
import glob
import time
import math
import numpy as np
import skvideo.io
import tensorflow as tf
import tensorlayer as tl
from model_tl import discriminator_I, discriminator_V,generator_I, getGRU
import torch

seed = 0
torch.manual_seed(seed)
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
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        video = torch.Tensor(trim(video))
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

dis_i = discriminator_I(nc, ndf)
dis_v = discriminator_V(nc, ndf,T)
gen_i = generator_I(nc, ngf, nz)
gru = getGRU(d_E, hidden_size)

''' prepare for train '''
label = tf.constant(0,tf.float32)

def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models')
def checkpoint(model, optimizer, epoch):
    filename = os.path.join(trained_path, '%s_epoch-%d' % (model.__class__.__name__, epoch))
    torch.save(model.state_dict(), filename + '.model')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)

''' setup optimizer '''
lr = 0.0002
optim_di = tf.optimizers.Adam(lr,0.5,0.999)
optim_dv = tf.optimizers.Adam(lr,0.5,0.999)
optim_gi = tf.optimizers.Adam(lr,0.5,0.999)
optim_gru = tf.optimizers.Adam(lr,0.5,0.999)

''' calculate grad of models '''
#我不会用tf写自动求导，以下是pytorch版本
'''
def bp_i(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_i(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    return err.data[0], outputs.data.mean()

def bp_v(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_v(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    return err.data[0], outputs.data.mean()
'''


''' generate input noise for fake vedio '''

def gen_z(n_frames):
    z_C = tf.Variable(tf.random.normal((batch_size,d_C)))
    z_C = z_C.unsqueeze(1).repeat(1,n_frames,1)
    eps = tf.Variable(tf.random.normal((batch_size, d_E)))
    gru.initHidden(batch_size)
    z = tf.concat(2,(z_M,z_C))
    return z.reshape((batch_size, n_frames, nz, 1, 1))

''' train models '''

for epoch in range(1, n_iter+1):
    ''' prepare real images '''
    real_videos = random_choice()
    real_videos = tf.Variable(real_videos)
    real_img = real_videos[:,:,np.random.randint(0,T),:,:]

    ''' prepare fake images '''
    n_frames = video_lengths[np.random.randint(0,n_videos)]
    Z = gen_z(n_frames)
    Z = trim_noise(Z)
    #generate videos
    Z = Z.reshape((batch_size*T,nz,1,1))
    fake_videos = gen_i(Z)
    fake_videos = fake_videos.reshape((batch_size,T,nc,img_size,img_size))
    # transpose => (batch_size, nc, T, img_size, img_size)
    fake_videos = fake_videos.transpose(2,1)
    #sample image
    fake_img = fake_videos[:,:,np.random.randint(0,T),:,:]

    ''' train discriminators '''
    #video
    dis_v
    #image

    ''' train generator '''

    if epoch % 100 == 0:
        print('[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f'
              % (epoch, n_iter, timeSince(start_time), err_Di, err_Dv, err_Gi, err_Gv, Di_real_mean, Di_fake_mean, Dv_real_mean, Dv_fake_mean))

    if epoch % 1000 == 0:
        save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)