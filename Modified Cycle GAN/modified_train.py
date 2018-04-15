import itertools
import pickle
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import matplotlib.pyplot as plt

from CycleGAN.models import Encoder
from CycleGAN.models import Decoder
from CycleGAN.models import DiscriminatorNew
from CycleGAN.utils import ReplayBuffer
from CycleGAN.utils import LambdaLR
from CycleGAN.utils import weights_init_normal
from CycleGAN.dataset import ImageDataset


start_epoch = 0
n_epochs = 200
decay_epoch = 100
batch_size = 1
data_root = './summer2winter_yosemite' # Change this to appropriate path to the dataset
lr = 0.0002
image_size = 128
input_nc = 3
output_nc = 3
activate_cuda = True
n_cpu = 8


if torch.cuda.is_available() and not activate_cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
encoder = Encoder(input_nc=input_nc)
decoder_A2B = Decoder(output_nc=output_nc)
decoder_B2A = Decoder(output_nc=output_nc)
netD_A = DiscriminatorNew(input_nc)
netD_B = DiscriminatorNew(output_nc)
loss_GAN_hist = []
loss_cycle_hist = []
loss_D_hist = []

if activate_cuda:
    encoder.cuda()
    decoder_A2B.cuda()
    decoder_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

if os.path.isfile('output modified/encoder.pth'):
    encoder.load_state_dict('output modified/encoder.pth')
    decoder_A2B.load_state_dict(torch.load('output modified/decoder_A2B.pth'))
    decoder_B2A.load_state_dict(torch.load('output modified/decoder_B2A.pth'))
    netD_A.load_state_dict(torch.load('output modified/netD_A.pth'))
    netD_B.load_state_dict(torch.load('output modified/netD_B.pth'))
else:
    encoder.apply(weights_init_normal)
    decoder_A2B.apply(weights_init_normal)
    decoder_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder_A2B.parameters(),
                                               decoder_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if activate_cuda else torch.Tensor
input_A = Tensor(batch_size, input_nc, image_size, image_size)
input_B = Tensor(batch_size, output_nc, image_size, image_size)
target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Resize(int(image_size), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(data_root, transforms_=transforms_, unaligned=True),
                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)


# Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))
###################################


###### Training ######
for epoch in range(start_epoch, n_epochs):
    loss_GAN_iter = 0
    loss_cycle_iter = 0
    loss_D_iter = 0
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # same_B = netG_A2B(real_B)
        # loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        # same_A = netG_B2A(real_A)
        # loss_identity_A = criterion_identity(same_A, real_A) * 5.0


        # GAN loss
        fake_B = decoder_A2B(encoder(real_A))
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = decoder_B2A((encoder(real_B)))
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = decoder_B2A(encoder(fake_B))
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = decoder_A2B(encoder(fake_A))
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        # output1, output2 = siamese_D_B(recovered_B, real_B)
        # loss_cycle_BAB = F.pairwise_distance(output1, output2) * 10.0

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        loss_GAN_iter += loss_GAN_A2B.data + loss_GAN_B2A.data
        loss_cycle_iter += loss_cycle_ABA.data + loss_cycle_BAB.data
        loss_D_iter += loss_D_A.data + loss_D_B.data
        print('epoch [%d/%d], iteration [%d/%d], loss_G: %.3f, loss_G_GAN: %.3f, loss_G_cycle: %.3f, loss_D: %.3f'
              % (epoch+1, n_epochs, i+1, len(dataloader), loss_G,
                 loss_GAN_A2B + loss_GAN_B2A, loss_cycle_ABA + loss_cycle_BAB, loss_D_A + loss_D_B))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    loss_GAN_hist.append(loss_GAN_iter/len(dataloader))
    loss_cycle_hist.append(loss_cycle_iter/len(dataloader))
    loss_D_hist.append(loss_cycle_iter/len(dataloader))

    # Save models checkpoints
    torch.save(encoder.state_dict(), 'output modified/encoder.pth')
    torch.save(decoder_A2B.state_dict(), 'output modified/encoder_A2B.pth')
    torch.save(decoder_B2A.state_dict(), 'output modified/encoder_B2A.pth')
    torch.save(netD_A.state_dict(), 'output modified/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output3/netD_B.pth')
    ###################################

