import torch.nn as nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
import matplotlib.pyplot as plt
import numpy as np
import os

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.noise_to_image = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.noise_to_image(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminate = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x).view(-1, 1).squeeze(1)

batch_size = 128
fixed_noise = Variable(torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1).cuda())


def train_network(discriminator, generator, dataloader):
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    y_real, y_fake = Variable(torch.ones(batch_size).cuda()), Variable(
        torch.zeros(batch_size).cuda())

    criterion = nn.BCELoss()
    criterion.cuda()

    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            real_data, _ = data
            z_ = torch.rand((batch_size, 100, 1, 1)).normal_(0, 1)
            real_data, z_ = Variable(real_data.cuda()), Variable(z_.cuda())

            discriminator.zero_grad()
            d_real_output = discriminator(real_data)
            d_real_loss = - torch.mean(d_real_output)

            fake_image = generator(z_)
            d_fake_output = discriminator(fake_image.detach())
            d_fake_loss = torch.mean(d_fake_output)

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            d_optimizer.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # if (i+1) % 5 == 0:
            generator.zero_grad()
            fake_image = generator(z_)
            d_fake_output = discriminator(fake_image)
            g_loss = -torch.mean(d_fake_output)

            g_loss.backward()
            g_optimizer.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, 25, i, len(dataloader),
                         d_loss.data[0], g_loss.data[0]))
            if i % 100 == 0:
                # vutils.save_image(real_cpu,
                #                   '%s/real_samples.png' % opt.outf,
                #                   normalize=True)
                fake = generator(fixed_noise)
                if not os.path.exists('WGAN output'):
                    os.makedirs('WGAN output')
                vutils.save_image(fake.data,
                                  '%s/fake_samples_epoch_%03d.png' % ('WGAN output', epoch),
                                  normalize=True)


if __name__ == '__main__':
    dataset = dset.LSUN(db_path='./data', classes=['church_outdoor_train'],
                        transform=transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    netG = Generator()
    netD = Discriminator()

    netG.apply(weights_init)
    netD.apply(weights_init)

    netG.cuda()
    netD.cuda()

    train_network(netD, netG, dataloader)

    # dataiter = iter(dataloader)
    # image, _ = dataiter.next()
    #
    # imshow(image[0])
