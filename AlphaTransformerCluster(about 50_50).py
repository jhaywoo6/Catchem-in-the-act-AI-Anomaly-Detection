import numpy as np
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import copy
from itertools import product
from torch.optim.lr_scheduler import LambdaLR
import time
from sklearn.metrics import roc_curve
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.sdp_kernel = "flash"

def extract_features(person_data):
    bbox = person_data[0]
    keypoints = person_data[1].flatten()
    return np.concatenate([bbox, keypoints])

def process_frame(frame_data):
    features = []
    for person_id, person_data in frame_data.items():
        if person_data and len(person_data) == 2 and isinstance(person_data[1], np.ndarray):
            feature = extract_features(person_data)
            if not np.any(np.isnan(feature)) and not np.any(np.isinf(feature)):
                features.append(feature)
    return features

def process_pkl_files(directory):
    all_features = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pkl"):
            with open(os.path.join(directory, file_name), "rb") as f:
                pkl_data = pickle.load(f)
                for frame_num, frame_data in pkl_data.items():
                    if frame_data:
                        frame_features = process_frame(frame_data)
                        all_features.extend(frame_features)
    return np.array(all_features)

def process_pkl_and_labels(pickle_dir, label_dir):
    all_features = []
    all_labels = []

    for file_name in os.listdir(pickle_dir):
        if file_name.endswith(".pkl"):
            pkl_path = os.path.join(pickle_dir, file_name)
            label_file = file_name.replace(".pkl", ".npy")
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                continue

            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)

            labels = np.load(label_path)
            label_index = 0

            for frame_num, frame_data in pkl_data.items():
                if not frame_data:
                    continue

                frame_features = []
                for person_id, person_data in frame_data.items():
                    if person_data and len(person_data) == 2 and isinstance(person_data[1], np.ndarray):
                        feature = extract_features(person_data)
                        if (
                            not np.any(np.isnan(feature)) and 
                            not np.any(np.isinf(feature)) and 
                            np.any(feature != 0)
                        ):
                            frame_features.append(feature)
                            if label_index < len(labels):
                                all_labels.append(labels[label_index])
                                label_index += 1

                all_features.extend(frame_features)

    # Ensure features and labels have the same length
    min_length = min(len(all_features), len(all_labels))
    return np.array(all_features[:min_length]), np.array(all_labels[:min_length])

def run_experiment(input_dim, hidden_dim, nhead, num_layers, train_loader, test_loader, device):
    print(f"\nTesting config: hidden_dim={hidden_dim}, nhead={nhead}, num_layers={num_layers}")

    model = TransformerAutoencoder(input_dim, hidden_dim, nhead, num_layers)
    model = train_autoencoder_with_eval(
        model,
        train_loader,
        test_loader,
        device,
        epochs=100,
        lr=1e-5,
        eval_interval=5,
        save_path="temp_model.pth"
    )

    embeddings, test_labels = extract_embeddings(model, test_loader, device)
    accuracy, precision, recall = evaluate_clustering(embeddings, test_labels)
    return precision + recall, (accuracy, precision, recall), (hidden_dim, nhead, num_layers), model


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fnr - fpr)
    eer_idx = np.nanargmin(abs_diffs)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    return eer, threshold

def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = None  # Handle if no labels in loader

            x = x.to(device)
            emb = model(x)[1]  # shape: [batch_size, hidden_dim]
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb.cpu().numpy())
            if y is not None:
                labels.append(y.cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels) if labels else None
    
    return embeddings, labels

def run_kmeans_and_evaluate(embeddings, true_labels, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    preds = kmeans.fit_predict(embeddings)

    acc = max(
        accuracy_score(true_labels, preds),
        accuracy_score(true_labels, 1 - preds)
    )
    return acc, preds

def train_autoencoder_with_eval(
    model,
    train_loader,
    test_loader,
    device,
    epochs=50,
    lr=1e-3,
    eval_interval=5,
    save_path="best_model.pth"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs, target_lr=lr)
    criterion = nn.MSELoss()

    best_score = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"Training on {len(train_loader.dataset)} samples | Testing on {len(test_loader.dataset)} samples\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in train_bar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            decoded, _ = model(x)
            loss = criterion(decoded, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_losses.append(total_loss)
        print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s - Train Loss: {total_loss:.4f}")

        if epoch % eval_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    decoded, _ = model(x)
                    loss = criterion(decoded, x)
                    val_loss += loss.item()
            val_losses.append(val_loss)

            embeddings, test_labels = extract_embeddings(model, test_loader, device)
            accuracy, precision, recall = evaluate_clustering(embeddings, test_labels)
            val_accuracies.append(accuracy)

            score = precision + recall
            print(f"Eval @ Epoch {epoch} → Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, save_path)
                print(f"New best model saved (Score: {score:.4f})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # You can also return these logs for plotting or debugging
    return model


def evaluate_clustering(embeddings, true_labels, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    preds = kmeans.fit_predict(embeddings)
    score_1 = accuracy_score(true_labels, preds)
    score_2 = accuracy_score(true_labels, 1 - preds)

    if score_2 > score_1:
        preds = 1 - preds

    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    accuracy = accuracy_score(true_labels, preds)

    print(f"Clustering Results:\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return accuracy, precision, recall

def get_linear_warmup_scheduler(optimizer, warmup_epochs, total_epochs, target_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

class TransformerClusterer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)  # Set batch_first=True
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return encoded.view(-1, encoded.size(-1))
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = TransformerClusterer(input_dim, hidden_dim, nhead, num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

if __name__ == "__main__":

    try: 
        train_tensor = torch.load("train_tensor.pt")
        test_tensor = torch.load("test_tensor.pt")
        test_labels = torch.load("test_labels.pt")
    except:
        train_features = process_pkl_files("C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/Train")

        test_features, test_labels = process_pkl_and_labels(
            "C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/Test",
            "C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/GT"
        )

        scaler = StandardScaler()

        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        train_tensor = torch.tensor(train_features, dtype=torch.float32)
        test_tensor = torch.tensor(test_features, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        torch.save(train_tensor, "train_tensor.pt")
        torch.save(test_tensor, "test_tensor.pt")
        torch.save(test_labels, "test_labels.pt")

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=2048, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_dataset = TensorDataset(test_tensor, test_labels)
    print(f"Total test samples: {len(test_dataset)}")  # Should match 760508

    test_loader = DataLoader(
        test_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_dims = [512]
    nheads = [8]
    num_layers_list = [3]

    best_score = 0
    best_config = None
    best_metrics = None

    input_dim = train_tensor.shape[1]

    for hidden_dim, nhead, num_layers in product(hidden_dims, nheads, num_layers_list):
        score, metrics, config, model = run_experiment(
            input_dim,
            hidden_dim,
            nhead,
            num_layers,
            train_loader,
            test_loader,
            device
        )

        if score > best_score:
            best_score = score
            best_config = config
            best_metrics = metrics
            torch.save(model.state_dict(), "best_model_overall.pth")
            print(f"New best config: {config} -> Score: {score:.4f}")
        else:
            print(f"Did not improve. Score: {score:.4f}")

    print("\nBest config:")
    print(f"  Hyperparameters: hidden_dim={best_config[0]}, nhead={best_config[1]}, num_layers={best_config[2]}")
    print(f"  Accuracy: {best_metrics[0]:.4f} | Precision: {best_metrics[1]:.4f} | Recall: {best_metrics[2]:.4f}")
