import torch
from torch.utils.data import DataLoader

def evaluate(encoder, classifier, dataset, device='cpu', batch_size=64):
    """
    Evaluate accuracy on a single dataset.
    """
    encoder.eval()
    classifier.eval()
    all_preds, all_labels = [], []

    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z = encoder(x)
            logits = classifier(z)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return (all_preds == all_labels).float().mean().item()


def evaluate_all_domains(encoder, classifier, load_dataset_fn, device='cpu'):
    """
    Evaluate model on all 10 evaluation domains after each domain training step.
    Returns accuracy matrix of shape (10, 10)
    """
    accuracies = torch.zeros(10, 10)
    encoder.to(device)
    classifier.to(device)

    # After domain 1
    print(" Evaluating after training D1")
    for j in range(1, 11):
        eval_dataset = load_dataset_fn(j, train=False)
        acc = evaluate(encoder, classifier, eval_dataset, device=device)
        accuracies[0, j - 1] = acc
        print(f"Eval after D1 → Eval D{j}: {acc:.2%}")

    # After domains 2–10
    for i in range(2, 11):
        print(f"\n🔍 Evaluating after training up to D{i}")
        for j in range(1, 11):
            eval_dataset = load_dataset_fn(j, train=False)
            acc = evaluate(encoder, classifier, eval_dataset, device=device)
            accuracies[i - 1, j - 1] = acc
            print(f"Eval after D{i} → Eval D{j}: {acc:.2%}")

    print("\nAccuracy Matrix (rows: after D1–D10, cols: Eval D1–D10):")
    print((accuracies * 100).round(decimals=2))
    return accuracies
