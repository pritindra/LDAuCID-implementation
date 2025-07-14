
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from utils.distance_utils import compute_prototypes, sliced_wasserstein_distance

def continual_adaptation(encoder, classifier, optimizer, memory_X, memory_y, gmm, load_dataset, generate_pseudo_data,
                         num_classes=10, latent_dim=64, memory_size_per_class=20, num_epochs=5, batch_size=64, tau=0.8, lambda_swd=1.0):

    for t in range(2, 11):
        print(f"\n Adapting to domain {t}")
        dt_dataset = load_dataset(t, train=True)
        dt_loader = DataLoader(dt_dataset, batch_size=batch_size, shuffle=True)

        Z_pseudo, y_pseudo = generate_pseudo_data(gmm, classifier, tau=tau, num_samples=1000)

        for epoch in range(num_epochs):
            for x_t in dt_loader:
                if isinstance(x_t, tuple):
                    x_t = x_t[0]

                mem_idx = random.sample(range(len(memory_X)), min(batch_size, len(memory_X)))
                pseudo_idx = random.sample(range(len(Z_pseudo)), min(batch_size, len(Z_pseudo)))
                X_mem = memory_X[mem_idx]; y_mem = memory_y[mem_idx]
                Z_mem = Z_pseudo[pseudo_idx]; y_pseudo_batch = y_pseudo[pseudo_idx]

                if Z_mem.size(0) == 0 or x_t.size(0) == 0:
                    continue

                feats_t = encoder(x_t)
                feats_mem = encoder(X_mem)
                logits_mem = classifier(feats_mem)
                logits_pseudo = classifier(Z_mem)
                loss_ce = F.cross_entropy(logits_mem, y_mem) + F.cross_entropy(logits_pseudo, y_pseudo_batch)
                swd = sliced_wasserstein_distance(feats_t, Z_mem)
                loss = loss_ce + lambda_swd * swd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        encoder.eval()
        all_feats, all_preds = [], []
        with torch.no_grad():
            for x_t in DataLoader(dt_dataset, batch_size=batch_size):
                if isinstance(x_t, tuple):
                    x_t = x_t[0]
                feats = encoder(x_t)
                preds = torch.argmax(classifier(feats), dim=1)
                all_feats.append(feats)
                all_preds.append(preds)

        all_feats = torch.cat(all_feats)
        all_preds = torch.cat(all_preds)

        prototypes = compute_prototypes(encoder(memory_X), memory_y, num_classes)
        for c in range(num_classes):
            idx_c = (all_preds == c).nonzero(as_tuple=True)[0]
            if len(idx_c) > 0:
                dists = torch.norm(all_feats[idx_c] - prototypes[c], dim=1)
                top_idx = idx_c[torch.argsort(dists)[:memory_size_per_class]]
                new_imgs = [
                    torch.from_numpy(dt_dataset.data[i]).permute(2, 0, 1).float() / 255.0
                    for i in top_idx
                ]
                new_imgs_tensor = torch.stack(new_imgs)
                memory_X = torch.cat([memory_X, new_imgs_tensor])
                memory_y = torch.cat([memory_y, torch.full((len(top_idx),), c, dtype=torch.long)])

        with torch.no_grad():
            feats_mem_total = encoder(memory_X)
        gmm.fit(feats_mem_total, memory_y)
