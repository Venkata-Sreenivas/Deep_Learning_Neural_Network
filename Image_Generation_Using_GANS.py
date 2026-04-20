import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Output a probability
        )

    def forward(self, img):
        return self.model(img)
latent_dim = 100
lr = 0.0002
batch_size = 128
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

fixed_noise = torch.randn(16, latent_dim).to(device)
print("Starting Training Loop...")

for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        current_batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)
        opt_d.zero_grad()
        outputs_real = discriminator(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        outputs_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_d.step()
        opt_g.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        opt_g.step()
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    if (epoch + 1) % 10 == 0 or epoch == 0:
        generator.eval() # Switch to evaluation mode
        with torch.no_grad():
            generated_imgs = generator(fixed_noise).cpu().view(-1, 28, 28)

            fig, axes = plt.subplots(4, 4, figsize=(5, 5))
            for j, ax in enumerate(axes.flatten()):
                # Rescale images from [-1, 1] back to [0, 1] for matplotlib
                img = (generated_imgs[j] + 1) / 2
                ax.imshow(img, cmap='gray')
                ax.axis('off')

            plt.suptitle(f"Epoch {epoch+1}")
            plt.show()
        generator.train() # Switch back to training mode
