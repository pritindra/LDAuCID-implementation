# models/gmm.py

import torch

class LatentGMM:
    def __init__(self, num_classes, latent_dim):
        self.num_classes = num_classes
        self.means = torch.zeros(num_classes, latent_dim)
        self.covariances = torch.zeros(num_classes, latent_dim, latent_dim)
        self.weights = torch.zeros(num_classes)

    def fit(self, features, labels):
        N = len(labels)
        for c in range(self.num_classes):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            Nc = len(idx)
            if Nc > 0:
                feats_c = features[idx]
                self.means[c] = feats_c.mean(dim=0)
                centered = feats_c - self.means[c]
                cov = centered.t() @ centered / (Nc + 1e-6)
                self.covariances[c] = cov + 1e-6 * torch.eye(centered.size(1))
                self.weights[c] = Nc / N
            else:
                self.means[c].zero_()
                self.covariances[c] = torch.eye(features.size(1))
                self.weights[c] = 1.0 / self.num_classes

    def sample(self, n_samples):
        if self.num_classes == 1:
            comp = torch.zeros(n_samples, dtype=torch.long)
        else:
            comp = torch.multinomial(self.weights, num_samples=n_samples, replacement=True)

        Z = torch.zeros(n_samples, self.means.size(1))
        for i, c in enumerate(comp):
            mean = self.means[c]
            cov = self.covariances[c]
            L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.size(0)))
            z = mean + L @ torch.randn(cov.size(0))
            Z[i] = z
        return Z

def generate_pseudo_data(gmm, classifier, tau=0.8, num_samples=1000):
    """
    Draw latent samples from the internal GMM and assign pseudo-labels
    using the classifier. Return confident latent vectors and their labels.
    """
    Z = gmm.sample(num_samples)  # (num_samples, latent_dim)
    with torch.no_grad():
        logits = classifier(Z)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        mask = conf > tau
    return Z[mask], pred[mask]
