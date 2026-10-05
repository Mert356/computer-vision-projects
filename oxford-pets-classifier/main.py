import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from dataset import Pets, check_no_overlap, eval_transform, read_manifest, train_transform
from model import PetCNN
from predict import get_device, predict_image
from train import evaluate, plot_history, plot_samples, train


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"  Accuracy:          {metrics['accuracy']:.2%}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"  Cat recall:        {metrics['cat_recall']:.2%}")
    print(f"  Dog recall:        {metrics['dog_recall']:.2%}")
    print(f"  Loss:              {metrics['loss']:.4f}")

def save_results(history, test_metrics, model, device):
    results = config.RESULTS_DIR
    results.mkdir(exist_ok=True)

    (results / 'history.json').write_text(json.dumps(history, indent=2))
    (results / 'test_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    plot_history(history, results / 'learning_curves.png')

    samples = []
    for relative in config.SAMPLE_IMAGES:
        path = config.DATA_DIR / relative
        score = predict_image(model, path, device)
        prediction = 'dog' if score >= 0.5 else 'cat'
        label = Path(relative).parent.name

        samples.append({'path': path, 'label': label, 'prediction': prediction, 'dog_score': score})
    plot_samples(samples, results / 'samples.png')
    print(f'\nResults saved to {results}')



def main():
    torch.manual_seed(config.SEED)
    device = get_device()

    splits = {}
    for split in ('train', 'val', 'test'):
        splits[split] = read_manifest(split)
    check_no_overlap(splits)

    train_data = Pets(config.DATA_DIR, splits['train'], train_transform)
    val_data = Pets(config.DATA_DIR, splits['val'], eval_transform)
    test_data = Pets(config.DATA_DIR, splits['test'], eval_transform)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    labels = [r['label'] for r in splits['train']]
    class_counts = [labels.count(0), labels.count(1)]
    print(f'Device: {device} | Train cats/dogs: {class_counts}')

    model = PetCNN().to(device)



    # pos_weight scales the loss of dog images by cats / dogs (about 0.48) so the rarer cats are not ignored
    pos_weight = torch.tensor(class_counts[0] / class_counts[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)


    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics(f'Test results on {len(test_data)} images', test_metrics)
    save_results(history, test_metrics, model, device)


if __name__ == '__main__':
    main()
