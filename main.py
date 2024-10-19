import torch
import torchvision
import Net
import numpy as np
from PIL import Image
import imageio

LATENT_DIM = 16
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
HIDDEN_DIMS = [512, 256, 128]

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def kl_divergence(mu, logvar):
    # KL Divergence between q(z|x) and p(z) ~ N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def train():

    data = torchvision.datasets.FashionMNIST(root="/data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    VAE = Net.VAE(784, HIDDEN_DIMS, LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=LR)

    BCE_loss = torch.nn.BCELoss(reduction='sum')

    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            images = images.view(images.shape[0], -1)
            labels = labels.to(device)

            reconstructed_images, mu, logvar  = VAE(images)
            rec_loss = BCE_loss(reconstructed_images, images)
            kl_loss = kl_divergence(mu, logvar)

            loss = rec_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch[{}/{}], Step[{}/{}], Rec Loss: {:.4f}, KL Loss: {:.4f}, Total Loss: {:.4f}'.format(
                    epoch + 1, EPOCHS, i + 1, len(loader), rec_loss.item(), kl_loss.item(), loss.item()
                )
            )

    torch.save(VAE.state_dict(), 'model.pth')

def interfirance():
    VAE = Net.VAE(784, HIDDEN_DIMS, LATENT_DIM).to(device)
    VAE.load_state_dict(torch.load('model.pth'))
    
    VAE.eval()

    for i in range(10):
        with torch.no_grad():
            sample = torch.randn(1, LATENT_DIM).to(device)
            generated_images = VAE.decoder(sample).cpu().view(-1, 1, 28, 28)
            torchvision.utils.save_image(generated_images, f"generated_images_{i}.png")

def generate_GIF(num_steps=100, image_size=(28, 28)):
        
    # generate two random labels
    label1 = np.random.randint(0, 10)
    label2 = np.random.randint(0, 10)
    while label1 == label2:
        label2 = np.random.randint(0, 10)
    
    dataset = torchvision.datasets.FashionMNIST(root="/data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    VAE = Net.VAE(784, HIDDEN_DIMS, LATENT_DIM).to(device)
    VAE.load_state_dict(torch.load('model.pth'))
    
    VAE.eval()

    # Find two images with the specified labels
    img1, img2 = None, None
    for img, label in dataset:
        if label == label1 and img1 is None:
            img1 = img.to(device).unsqueeze(0)  # Add batch dimension and move to device
        if label == label2 and img2 is None:
            img2 = img.to(device).unsqueeze(0)  # Add batch dimension and move to device
        if img1 is not None and img2 is not None:
            break
    

    # Ensure we found both images
    assert img1 is not None and img2 is not None, f"Could not find both labels {fashion_mnist_labels[label1]} and {fashion_mnist_labels[label2]} in the dataset."
    
    label1_name = fashion_mnist_labels[label1].replace("/", "-")  # Replace any '/' in the label with '-'
    label2_name = fashion_mnist_labels[label2].replace("/", "-")
    gif_filename = f"transform_from_{label1_name}_to_{label2_name}.gif"

    # Pass images through the encoder to get latent vectors z1 and z2
    with torch.no_grad():
        print(img1.shape)
        z1, _ = VAE.encoder(img1.view(1, -1))
        z2, _ = VAE.encoder(img2.view(1, -1))

    interpolation_vectors = np.linspace(z1.cpu().numpy(), z2.cpu().numpy(), num_steps)

    # Create a list to store generated images
    images = []

    # Traverse the latent space
    for i, z in enumerate(interpolation_vectors):
        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device) # Convert to tensor and add batch dimension        
        with torch.no_grad():
            # Decode the latent vector into an image
            generated_image = VAE.decoder(z_tensor).cpu().squeeze(0)       # Remove batch dimension
        # Transform tensor to an image
        img = (generated_image.numpy().reshape(28, 28) * 255).astype(np.uint8)  # Convert to [0, 255] scale
        img = Image.fromarray(img).resize(image_size)                  # Resize to the desired size
        
        # Append image to the list
        images.append(img)

    # Save images as a GIF
    images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f"GIF saved at {gif_filename}")


def main():
    
    
    generate_GIF()

if __name__ == '__main__':
    main()