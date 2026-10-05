import matplotlib
import torch
from PIL import Image

import config

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_metrics(total_loss, targets, predictions):
    cats = targets == 0
    dogs = targets == 1
    cat_recall = (predictions[cats] == 0).float().mean().item()
    dog_recall = (predictions[dogs] == 1).float().mean().item()

    accuracy = (predictions == targets).float().mean().item()
    return {
        'loss': total_loss / len(targets),
        'accuracy': accuracy,
        'balanced_accuracy': (cat_recall + dog_recall) / 2,
        'cat_recall': cat_recall,
        'dog_recall': dog_recall
    }


def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            all_targets.append(targets.cpu())
            all_predictions.append((logits >= 0).float().cpu())

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    return compute_metrics(total_loss, targets, predictions)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    history = []
    best_score = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        all_targets, all_predictions = [], []

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            all_targets.append(targets.cpu())

            predictions = (logits >= 0).float()
            all_predictions.append(predictions.cpu())


        train_metrics = compute_metrics(total_loss, torch.cat(all_targets), torch.cat(all_predictions))
        val_metrics = evaluate(model, val_loader, criterion, device)

        score = val_metrics['balanced_accuracy']
        lr = optimizer.param_groups[0]['lr']



        history.append({'epoch': epoch, 'lr': lr, 'train': train_metrics, 'val': val_metrics})
        scheduler.step(score)


        print(
            f"Epoch {epoch:02d} | Train loss: {train_metrics['loss']:.4f} | Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.2%} | Balanced acc: {score:.2%} | "
            f"Cat: {val_metrics['cat_recall']:.2%} | Dog: {val_metrics['dog_recall']:.2%} | LR: {lr:.2g}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    best_weights = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_weights)
    print(f'\nBest validation balanced accuracy: {best_score:.2%}')
    return history




def plot_history(history, path):
    epochs = [row['epoch'] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for split in ('train', 'val'):
        axes[0].plot(epochs, [row[split]['loss'] for row in history], label=split)
        axes[1].plot(epochs, [row[split]['balanced_accuracy'] for row in history], label=split)
    axes[0].set_ylabel('Class-weighted BCE')
    axes[1].set_ylabel('Balanced accuracy')
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.legend()
        axis.grid(alpha=0.2)


    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_samples(samples, path):
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4.4))

    for axis, sample in zip(axes, samples):
        with Image.open(sample['path']) as image:
            axis.imshow(image.convert('RGB'))

        score = sample['dog_score']
        prediction = sample['prediction']
        confidence = score if prediction == 'dog' else 1 - score
        title = f"True: {sample['label']} | Predicted: {prediction}\n{prediction} {confidence:.1%} (dog score {score:.3f})"
        axis.set_title(title, fontsize=10)
        axis.axis('off')

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


