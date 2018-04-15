import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from CycleGAN.models import Encoder
from CycleGAN.models import Decoder
from CycleGAN.dataset import ImageDataset

activate_cuda = True
input_nc = 3
output_nc = 3
batch_size = 1
image_size = 128
n_cpu = 8
data_root = './summer2winter_yosemite'# Change this to the appropriate path of the dataset
encoder_path = './output modified/encoder.pth'
decoder_A2B_path = './output modified/decoder_A2B.pth'
decoder_B2A_path = './output modified/decoder_B2A.pth'

if torch.cuda.is_available() and not activate_cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
encoder = Encoder(input_nc=input_nc)
decoder_A2B = Decoder(output_nc=output_nc)
decoder_B2A = Decoder(output_nc=output_nc)

if activate_cuda:
    encoder.cuda()
    decoder_A2B.cuda()
    decoder_B2A.cuda()

# Load state dicts
encoder.load_state_dict(torch.load(encoder_path))
decoder_A2B.load_state_dict(torch.load(decoder_A2B_path))
decoder_B2A.load_state_dict(torch.load(decoder_B2A_path))

# Set model's test mode
encoder.eval()
decoder_A2B.eval()
decoder_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if activate_cuda else torch.Tensor
input_A = Tensor(batch_size, input_nc, image_size, image_size)
input_B = Tensor(batch_size, output_nc, image_size, image_size)

# Dataset loader
transforms_ = [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(data_root, transforms_=transforms_, mode='test'),
                        batch_size=batch_size, shuffle=False, num_workers=n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output modified/A'):
    os.makedirs('output modified/A')
if not os.path.exists('output modified/B'):
    os.makedirs('output modified/B')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5 * (decoder_A2B(encoder(real_A)).data + 1.0)
    fake_A = 0.5 * (decoder_B2A(encoder(real_B)).data + 1.0 )

    # Save image files
    save_image(fake_A, 'output modified/A/%04d.png' % (i+1))
    save_image(fake_B, 'output modified/B/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################