import os
import sys
import time
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# Ensure root directory is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data import load_splits, build_vocab, encode, class_weights
from src.evaluate import evaluate_model, count_params, time_inference, save_results


class ComplaintDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_sizes, num_filters, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, n_classes)

    def forward(self, x):
        embedded = self.embedding(x)          # (batch_size, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(embedded))
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            pooled.append(p)

        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_val(model, dataloader, device):
    model.eval()
    all_preds, all_trues = [], []
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            val_loss += loss.item() * len(y_b)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(y_b.cpu().numpy())

    macro_f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    return val_loss / len(all_trues), macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train TextCNN on CFPB Complaints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/textcnn.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with split CSVs")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--test", action="store_true",
                        help="FINAL RUN ONLY: evaluate on the held-out test split")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Run Config] Seed: {args.seed} | Device: {device}")

    cfg = {
        'max_len': 224,
        'embed_dim': 128,
        'filter_sizes': [3, 4, 5],
        'num_filters': 100,
        'dropout': 0.5,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 0.01
    }
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg.update(yaml.safe_load(f))

    # 1. Load Data
    train_data, val_data, test_data, meta = load_splits(args.data_dir)
    n_classes = meta['n_classes']
    classes = meta['classes']

    # 2. Vocabulary & Encoding (fit on training data ONLY)
    vocab = build_vocab(train_data.text)
    pad_idx = vocab.get('<pad>', 0)

    X_train = encode(train_data.text, vocab, max_len=cfg['max_len'])
    X_val   = encode(val_data.text,   vocab, max_len=cfg['max_len'])

    train_loader = DataLoader(
        ComplaintDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(train_data.y, dtype=torch.long)),
        batch_size=cfg['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        ComplaintDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(val_data.y, dtype=torch.long)),
        batch_size=cfg['batch_size'],
        shuffle=False
    )

    # 3. Class-weighted Cross-Entropy Loss
    w = class_weights(train_data.y, n_classes)
    loss_weights = torch.tensor(w, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # 4. Model & AdamW Optimiser
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=cfg['embed_dim'],
        n_classes=n_classes,
        filter_sizes=cfg['filter_sizes'],
        num_filters=cfg['num_filters'],
        dropout=cfg['dropout'],
        pad_idx=pad_idx
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # 5. Training Loop with Early Stopping on Validation Macro F1
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    best_val_macro_f1 = -1.0
    best_weights = None
    patience_count = 0
    start_time = time.time()

    print("[Training] Commencing training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)

            optimiser.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * len(y_b)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss, epoch_val_f1 = evaluate_val(model, val_loader, device)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_macro_f1'].append(epoch_val_f1)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Macro F1: {epoch_val_f1:.4f}")

        if epoch_val_f1 > best_val_macro_f1:
            best_val_macro_f1 = epoch_val_f1
            best_weights = model.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[Early Stopping] Triggered at epoch {epoch} (Patience: {args.patience})")
                break

    train_time_s = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 6. Final evaluation.
    # The test split is used only with --test, for the single final run;
    # every tuning run is scored on validation.
    if args.test:
        eval_data, split_name = test_data, 'test'
        X_eval = encode(eval_data.text, vocab, max_len=cfg['max_len'])
        eval_loader = DataLoader(
            ComplaintDataset(torch.tensor(X_eval, dtype=torch.long),
                             torch.tensor(eval_data.y, dtype=torch.long)),
            batch_size=cfg['batch_size'],
            shuffle=False
        )
    else:
        eval_data, split_name, eval_loader = val_data, 'val', val_loader

    model.eval()
    val_preds, val_probas = [], []
    with torch.no_grad():
        for x_b, _ in eval_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probas = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probas, axis=1)
            val_probas.append(probas)
            val_preds.append(preds)

    y_pred = np.concatenate(val_preds, axis=0)
    y_proba = np.concatenate(val_probas, axis=0)

    res = evaluate_model(eval_data.y, y_pred, y_proba, classes, split=split_name)
    res['params'] = count_params(model)
    res['train_time_s'] = train_time_s

    def predict_fn(batch_text):
        model.eval()
        with torch.no_grad():
            x_enc = encode(batch_text, vocab, max_len=cfg['max_len'])
            x_t = torch.tensor(x_enc, dtype=torch.long).to(device)
            return F.softmax(model(x_t), dim=1).cpu().numpy()

    res['inference_ms_per_1k'] = time_inference(predict_fn, eval_data.text[:1000])

    os.makedirs('results', exist_ok=True)
    name = 'textcnn_test' if args.test else 'textcnn'
    save_results(name, res, history=history, config=cfg, seed=args.seed)
    print(f"[Saved] Output written to results/{name}_seed{args.seed}.json")


if __name__ == "__main__":
    main()
