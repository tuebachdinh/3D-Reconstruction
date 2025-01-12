import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_preprocessed import poses, images
device = torch.device("cpu")


class ProbabilisticNeRF(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=3, num_layers=4, dropout_rate=0.1):
        """
        A Bayesian NeRF model with Monte Carlo Dropout.

        Args:
            input_dim (int): Input dimension (3D position + view direction).
            hidden_dim (int): Hidden layer size.
            output_dim (int): Output dimension (RGB color).
            num_layers (int): Number of layers.
            dropout_rate (float): Dropout rate for uncertainty modeling.
        """
        super(ProbabilisticNeRF, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def train_bayesian_nerf(images, poses, H, W, num_epochs=1000, lr=1e-4, batch_size=1024, dropout_rate=0.1):
    """
    Train a Bayesian NeRF model (CPU mode).
    """
    # Initialize model, optimizer, and loss
    nerf = ProbabilisticNeRF(input_dim=6, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Flatten images for sampling
    N_images = images.shape[0]
    images_flat = images.reshape(N_images, -1, 3)  # (N_images, H*W, 3)

    for epoch in range(num_epochs):
        # Randomly select an image and pixels
        idx_image = torch.randint(0, N_images, (1,)).item()  # Random image index
        idx_pixels = torch.randint(0, H * W, (batch_size,))  # Random pixels

        # Select pixel colors and pose
        pixel_colors = images_flat[idx_image, idx_pixels].to(device)  # (batch_size, 3)
        pose = poses[idx_image].to(device)  # (4, 4)

        # Generate rays
        origins = pose[:3, 3].expand(batch_size, -1)  # (batch_size, 3)
        i, j = torch.div(idx_pixels, W, rounding_mode='floor'), idx_pixels % W
        directions = torch.stack([(i - W / 2) / W, -(j - H / 2) / H, -torch.ones_like(i)], dim=-1)
        directions = directions @ pose[:3, :3].T  # Rotate directions

        # Query Bayesian NeRF model
        inputs = torch.cat([origins, directions], dim=-1)  # (batch_size, 6)
        predicted_colors = nerf(inputs)

        # Compute loss
        loss = loss_fn(predicted_colors, pixel_colors)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

    return nerf


def estimate_uncertainty(nerf, inputs, num_samples=10):
    """
    Estimate uncertainty using Monte Carlo Dropout.

    Args:
        nerf (nn.Module): Trained Bayesian NeRF model.
        inputs (torch.Tensor): Input rays of shape (N, 6).
        num_samples (int): Number of stochastic forward passes.

    Returns:
        mean_prediction (torch.Tensor): Mean prediction of shape (N, 3).
        uncertainty (torch.Tensor): Variance of predictions of shape (N, 3).
    """
    predictions = torch.stack([nerf(inputs) for _ in range(num_samples)], dim=0)  # (num_samples, N, 3)
    mean_prediction = predictions.mean(dim=0)  # Mean prediction
    uncertainty = predictions.var(dim=0)  # Variance (uncertainty)
    return mean_prediction, uncertainty


def render_novel_view_with_uncertainty(nerf, pose, H, W, num_samples=10):
    """
    Render a novel view using Bayesian NeRF with uncertainty estimation (CPU mode).
    """
    pose = pose.to(device)
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    directions = torch.stack([(i - W / 2) / W, -(j - H / 2) / H, -torch.ones_like(i)], dim=-1)
    directions = (directions @ pose[:3, :3].T).reshape(-1, 3)  # Rotate directions
    origins = pose[:3, 3].expand(directions.shape[0], -1)  # Camera origin for each ray

    # Flatten rays
    inputs = torch.cat([origins, directions], dim=-1)  # (H * W, 6)

    # Estimate uncertainty
    mean_prediction, uncertainty = estimate_uncertainty(nerf, inputs, num_samples)

    # Reshape outputs
    mean_image = mean_prediction.reshape(H, W, 3).cpu()
    uncertainty_map = uncertainty.mean(dim=-1).reshape(H, W).cpu()  # Variance over RGB channels
    return mean_image, uncertainty_map


# Load data (replace this with your actual data loading logic)
H, W = 800, 800  # Image dimensions
num_epochs = 1000
lr = 1e-4
batch_size = 512

# Assuming `images` and `poses` are preloaded tensors
# images: (N_images, H, W, 3), poses: (N_images, 4, 4)
trained_nerf = train_bayesian_nerf(images.to(device), poses.to(device), H, W, num_epochs=num_epochs, lr=lr, batch_size=batch_size)
novel_pose = poses[0]
mean_image, uncertainty_map = render_novel_view_with_uncertainty(trained_nerf, novel_pose, H, W)

# Visualize mean image
plt.figure(figsize=(8, 8))
plt.imshow(mean_image.detach().numpy())
plt.title("Mean Prediction")
plt.axis("off")
plt.show()

# Visualize uncertainty map
plt.figure(figsize=(8, 8))
plt.imshow(uncertainty_map.detach().numpy(), cmap="hot")
plt.title("Uncertainty Heatmap")
plt.colorbar(label="Uncertainty")
plt.axis("off")
plt.show()
