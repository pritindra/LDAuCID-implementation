
import torch

# Utility: compute class prototypes (mean of features) given features and labels
def compute_prototypes(features, labels, num_classes):
    prototypes = []
    for c in range(num_classes):
        class_feats = features[labels == c]
        if len(class_feats) > 0:
            prototypes.append(class_feats.mean(dim=0))
        else:
            prototypes.append(torch.zeros(features.shape[1], device=features.device))
    return torch.stack(prototypes)  # shape: (num_classes, latent_dim)

# Utility: compute Euclidean distance between features and prototypes
def l2_distances(features, prototypes):
    # features: (batch, dim), prototypes: (num_classes, dim)
    # Returns (batch, num_classes) distances
    f_sq = (features ** 2).sum(dim=1, keepdim=True)
    p_sq = (prototypes ** 2).sum(dim=1)
    dist = f_sq + p_sq.unsqueeze(0) - 2 * features @ prototypes.t()
    return dist

# Compute Sliced Wasserstein Distance (SWD) between two sets of vectors
def sliced_wasserstein_distance(X, Y, num_projections=50, device='cpu'):
    # X, Y: tensors of shape (n_samples, feature_dim)
    d = X.size(1)
    swd = 0.0
    for _ in range(num_projections):
        theta = torch.randn(d, device=device)
        theta = theta / theta.norm()
        proj_X = X @ theta  # (n,)
        proj_Y = Y @ theta
        proj_X, _ = torch.sort(proj_X)
        proj_Y, _ = torch.sort(proj_Y)
        swd += torch.mean((proj_X - proj_Y) ** 2)
    return swd / num_projections
