import random
import torch
import torch.nn as nn
import gymnasium as gym
import argparse
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.utils as vutils
import time
import typing as tt
import numpy as np
import cv2
from gymnasium import spaces
import ale_py
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(old_space.low), self.observation(old_space.high), 
            dtype = np.float32
        )

    def observation(self, observation: gym.core.ObsType)->gym.core.ObsType :
        new_obs = cv2.resize(observation, (IMAGE_SIZE,IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

class Generator(nn.Module):
    '''
    takes as input a vector of random numbers (latent vector) and, by using the “transposed convolution” operation 
    (it is also known as deconvolution), converts this vector into a color image of the original resolution.
    '''
    def __init__(self,output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)
    


class Discriminator(nn.Module):
    '''
    takes our scaled color image as input and, by applying five layers of convolutions,
    converts it into a single number passed through a Sigmoid nonlinearity. 
    The output from Sigmoid is interpreted as the probability that Discriminator thinks our input 
    image is from the real dataset.
    '''
    def __init__(self,input_shape):
        super(Discriminator, self).__init__()
        self.conv_pipe = nn.Sequential(
             nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),

            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),

            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),

            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),

            nn.Sigmoid()
        )
    
    def forward(self,x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1,1).squeeze(dim=1)

def iterate_batches(envs:tt.List[gym.Env], batch_size:int=BATCH_SIZE)->tt.Generator[torch.Tensor,None,None]:
    '''
    This function infinitely samples the environment from the provided list, 
    issues random actions, and remembers observations in the batch list. 
    When the batch becomes of the required size, we normalize the image, convert it to a tensor, 
    and yield from the generator. The check for the non-zero mean of the observation is required due 
    to a bug in one of the games to prevent the flickering of an image.

    envs: a list of OpenAI Gym (or Gymnasium) environments.
    batch_size: number of samples per batch before yielding.
    The function is a Python generator that yields PyTorch tensors (batches of normalized observations)
    '''
    batch = [e.reset()[0] for e in envs]
    # Each environment e is reset, returning its initial observation (e.reset() usually returns (obs, info)).
    # The [0] extracts just the observation part.
    # So initially, batch contains one observation from each environment.

    env_gen = iter(lambda:random.choice(envs), None)
    # So each loop iteration will randomly pick one of the environments to step.

    while True:
        e = next(env_gen)
        action = e.action_space.sample()
        obs, reward, is_done, is_trunc, _ = e.step(action)
        # Only add the observation to the batch if its mean pixel/intensity value > 0.01.
        # This is likely a quick filter to skip "empty" or irrelevant frames (e.g., black screens)
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            # Normalising input to [-1..1] and convert to tensor
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)  #every yield gives you a tensor of shape (batch_size, *obs_shape) with normalized data.
            batch.clear()
        if is_done or is_trunc:
            e.reset()[0]




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='Device name, default=cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Register Atari environments
    gym.register_envs(ale_py)
    
    envs = [
        InputWrapper(gym.make(name)) 
        for name in ('ALE/Breakout-v5', 'ALE/AirRaid-v5', 'ALE/Pong-v5')
    ]
    shape = envs[0].observation_space.shape

    #create the networks and place them on the device
    net_discr = Discriminator(shape).to(device)
    net_gener = Generator(shape).to(device)

    objective = nn.BCELoss() #loss function for the descriminator ( real / fake )

    #optimizers for both the networks
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    
    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE,device=device)
    ts_start = time.time()
    writer = SummaryWriter()

    for batch_v in iterate_batches(envs):
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        #minibatch of real images
        batch_v = batch_v.to(device)

        #generator output on random input vectors
        gen_output_v = net_gener(gen_input_v)


        # train discriminator
        dis_optimizer.zero_grad()
        # We compute D's scores on real images and fake images.
        # we need to call the detach() function on the generator's output to prevent gradients of this training 
        # pass from flowing into the generator (detach() is a method of tensor, which makes a copy of it without 
        # connection to the parent's operation, i.e., detaching the tensor from the parent's graph).
        dis_output_true_v = net_discr(batch_v)                # D(x) on real images
        dis_output_fake_v = net_discr(gen_output_v.detach())  # D(G(z)) on fake images
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward() 
        dis_optimizer.step() 
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            dt = time.time() - ts_start
            log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
            ts_start = time.time()
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            img = vutils.make_grid(gen_output_v.data, normalize=True)
            writer.add_image("fake", img, iter_no)
            img = vutils.make_grid(batch_v.data, normalize=True)
            writer.add_image("real", img, iter_no)






