
import torch
from models.encoder import CNNEncoder
from models.classifier import Classifier
from models.gmm import LatentGMM, generate_pseudo_data
from utils.custom_dataset import CustomDataset
from core.training import train_source_domain
from core.replay import continual_adaptation
from core.eval import evaluate_all_domains

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Settings
latent_dim = 64
num_classes = 10

# Load dataset function
def load_dataset(domain_idx, train=True):
    path = f"dataset/dataset/part_one_dataset/{'train_data' if train else 'eval_data'}/{domain_idx}_{'train' if train else 'eval'}_data.tar.pth"
    labeled = (train and domain_idx == 1) or not train
    return CustomDataset(path, labeled=labeled)

# Initialize encoder and classifier
encoder = CNNEncoder(latent_dim=latent_dim).to(device)
classifier = Classifier(latent_dim=latent_dim, num_classes=num_classes).to(device)

# === Step 1: Train on Domain 1 === #
print("\nTraining on domain D1...")
d1_dataset = load_dataset(1, train=True)
memory_X, memory_y, feats, labels, optimizer = train_source_domain(
    encoder, classifier, d1_dataset, num_classes=num_classes, latent_dim=latent_dim
)

# === Step 2: Fit GMM === #
gmm = LatentGMM(num_classes=num_classes, latent_dim=latent_dim)
gmm.fit(feats, labels)

# === Step 3: Continual Adaptation from D2 to D10 === #
continual_adaptation(
    encoder, classifier, optimizer, memory_X, memory_y, gmm,
    load_dataset, generate_pseudo_data,
    num_classes=num_classes, latent_dim=latent_dim
)

# === Step 4: Evaluate on All Domains === #
print("\nFinal Evaluation on All Domains:")
accuracies = evaluate_all_domains(encoder, classifier, load_dataset, device=device)
