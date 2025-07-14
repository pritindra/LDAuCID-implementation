
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from utils.distance_utils import compute_prototypes, sliced_wasserstein_distance

def train_source_domain(encoder, classifier, dataset, num_classes, memory_size_per_class=20, latent_dim=64, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train for 3 epochs on source domain (D1)
    for epoch in range(3):
        encoder.train()
        classifier.train()
        for x, y in loader:
            z = encoder(x)
            logits = classifier(z)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract features for memory and GMM fitting
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(encoder(x))
            labels.append(y)
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    # Compute prototypes
    prototypes = compute_prototypes(feats, labels, num_classes)

    # Build memory from most representative samples (MoF)
    memory_X, memory_y = [], []
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            dists = torch.norm(feats[idx] - prototypes[c], dim=1)
            top_idx = idx[torch.argsort(dists)[:memory_size_per_class]]
            imgs = [
                torch.from_numpy(dataset.data[i]).permute(2, 0, 1).float() / 255.0
                for i in top_idx
            ]
            memory_X.append(torch.stack(imgs))
            memory_y.append(labels[top_idx])

    memory_X = torch.cat(memory_X)
    memory_y = torch.cat(memory_y)

    return memory_X, memory_y, feats, labels, optimizer
