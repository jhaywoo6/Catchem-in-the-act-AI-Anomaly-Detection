import numpy as np
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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
import random

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
    frame_features = {}
    for person_id, person_data in frame_data.items():
        if person_data and len(person_data) == 2 and isinstance(person_data[1], np.ndarray):
            feature = extract_features(person_data)
            if (
                not np.any(np.isnan(feature)) and 
                not np.any(np.isinf(feature)) and 
                np.any(feature != 0)  # Also ignore all-zero features
            ):
                frame_features[person_id] = feature
    return frame_features

def process_pkl_files(directory):
    person_sequences = {}

    for file_name in os.listdir(directory):
        if file_name.endswith(".pkl"):
            file_sequences = {}

            with open(os.path.join(directory, file_name), "rb") as f:
                pkl_data = pickle.load(f)

                for frame_num in sorted(pkl_data.keys()):
                    frame_data = pkl_data[frame_num]
                    if frame_data:
                        frame_features = process_frame(frame_data)
                        for person_id, feature in frame_features.items():
                            if person_id not in file_sequences:
                                file_sequences[person_id] = []
                            file_sequences[person_id].append(feature)

            # Merge file_sequences into overall person_sequences
            for person_id, sequence in file_sequences.items():
                if person_id not in person_sequences:
                    person_sequences[person_id] = []
                person_sequences[person_id].extend(sequence)

    return person_sequences

def process_pkl_and_labels(pickle_dir, label_dir):
    all_sequences = []
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

            person_sequences = {}

            for frame_num in sorted(pkl_data.keys()):
                frame_data = pkl_data[frame_num]
                if not frame_data:
                    continue

                for person_id, person_data in frame_data.items():
                    if person_data and len(person_data) == 2 and isinstance(person_data[1], np.ndarray):
                        feature = extract_features(person_data)
                        if (
                            not np.any(np.isnan(feature)) and 
                            not np.any(np.isinf(feature)) and 
                            np.any(feature != 0)
                        ):
                            if person_id not in person_sequences:
                                person_sequences[person_id] = []
                            person_sequences[person_id].append(feature)

            # After processing all frames in one file
            for person_id, sequence in person_sequences.items():
                if len(sequence) > 0 and label_index < len(labels):
                    all_sequences.append(np.array(sequence))  # sequence is list of features
                    all_labels.append(labels[label_index])
                    label_index += 1

    return all_sequences, np.array(all_labels)

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
            emb = model(x)[1]  # shape: [batch_size, seq_len, hidden_dim]
            emb = emb.mean(dim=1)  # Apply mean pooling over the sequence length
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
            torch.cuda.empty_cache()
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

            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            print(f"New best model saved (Score: {score:.4f})")
            torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # You can also return these logs for plotting or debugging
    return model

def evaluate_clustering(embeddings, true_labels, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    preds = kmeans.fit_predict(embeddings)

    # Compute metrics for both original and inverted predictions
    precision_1 = precision_score(true_labels, preds)
    recall_1 = recall_score(true_labels, preds)
    accuracy_1 = accuracy_score(true_labels, preds)

    precision_2 = precision_score(true_labels, 1 - preds)
    recall_2 = recall_score(true_labels, 1 - preds)
    accuracy_2 = accuracy_score(true_labels, 1 - preds)

    # Choose the better alignment
    if precision_2 > precision_1:
        preds = 1 - preds
        precision, recall, accuracy = precision_2, recall_2, accuracy_2
    else:
        precision, recall, accuracy = precision_1, recall_1, accuracy_1

    print(f"Clustering Results:\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return accuracy, precision, recall

def get_linear_warmup_scheduler(optimizer, warmup_epochs, total_epochs, target_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

def collate_fn(batch):
    if isinstance(batch[0], tuple):  # Check if the batch contains (sequence, label) tuples
        sequences, labels = zip(*batch)  # Unpack sequences and labels
        sequences = pad_sequence(sequences, batch_first=True)  # Pad sequences in the batch
        labels = torch.stack(labels)  # Stack labels into a tensor
        return sequences, labels
    else:  # Handle datasets without labels
        sequences = pad_sequence(batch, batch_first=True)  # Pad sequences in the batch
        return sequences
    
def qualitative_testing(model, test_features, test_labels, device, num_samples=10):
    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Randomly select a frame
            idx = random.randint(0, len(test_features) - 1)
            frame = test_features[idx].unsqueeze(0).to(device)  # Add batch dimension
            true_label = test_labels[idx].item()

            # Get the embedding
            _, embedding = model(frame)
            embedding = embedding.mean(dim=1).cpu().numpy()  # Mean pooling over sequence length

            # Simulate a binary classification decision (e.g., threshold-based or pre-trained clustering)
            pred_label = 1 if embedding.mean() > 0 else 0  # Example threshold-based decision

            # Append results
            results.append({
                "Person_ID": idx,
                "True_Label": true_label,
                "Predicted_Label": pred_label
            })

    # Print results
    for i, result in enumerate(results):
        print(f"Sample {i + 1}:")
        print(f"  Person ID: {result['Person_ID']}")
        print(f"  True Label: {result['True_Label']}")
        print(f"  Predicted Label: {result['Predicted_Label']}")
        print("-" * 30)

class TransformerClusterer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)  # Set batch_first=True
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        encoded = self.encoder(x)
        return encoded
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = TransformerClusterer(input_dim, hidden_dim, nhead, num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]



if __name__ == "__main__":

    try: 
        train_tensors = torch.load("train_tensor.pt")
        test_tensors = torch.load("test_tensor.pt")
        test_labels_tensor = torch.load("test_labels.pt")
        
    except:
        train_features = process_pkl_files("C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/Train")

        test_features, test_labels = process_pkl_and_labels(
            "C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/Test",
            "C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/GT"
        )

        scaler = StandardScaler()

        for key in train_features:
            train_features[key] = scaler.fit_transform(train_features[key])
        for i in range(len(test_features)):
            test_features[i] = scaler.transform(test_features[i])

        train_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in train_features.values()]
        test_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in test_features]
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

        torch.save(train_tensors, "train_tensor.pt")
        torch.save(test_tensors, "test_tensor.pt")
        torch.save(test_labels_tensor, "test_labels.pt")

    try:
        train_poselift = torch.load("train_poselift.pt")
        test_poselift = torch.load("test_poselift.pt")
        test_poselift_labels = torch.load("test_poselift_labels.pt")
    except:
        train_features_poselift = process_pkl_files("C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files_Pose_lift/Train")
        test_features_poselift, test_labels_poselift = process_pkl_and_labels(
            "C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files_Pose_lift/Test",
            "C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files_Pose_lift/GT"
        )

        scaler = StandardScaler()

        for key in train_features_poselift:
            train_features_poselift[key] = scaler.fit_transform(train_features_poselift[key])
        for i in range(len(test_features_poselift)):
            test_features_poselift[i] = scaler.transform(test_features_poselift[i])


        train_poselift = [torch.tensor(seq, dtype=torch.float32) for seq in train_features_poselift.values()]
        test_poselift = [torch.tensor(seq, dtype=torch.float32) for seq in test_features_poselift]
        test_poselift_labels  = torch.tensor(test_labels_poselift, dtype=torch.float32)

        torch.save(train_poselift, "train_poselift.pt")
        torch.save(test_poselift, "test_poselift.pt")
        torch.save(test_poselift_labels, "test_poselift_labels.pt")

    train_dataset = SequenceDataset(train_tensors)
    test_dataset = SequenceDataset(test_tensors, test_labels_tensor)
    print(f"Total test samples: {len(test_dataset)}")  # Should match 760508

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_dims = [64]
    nheads = [8]
    num_layers_list = [2]

    best_score = 0
    best_config = None
    best_metrics = None

    input_dim = train_tensors[0].shape[1]

    poselift_only_path = "PoseLiftOnly.pth"
    if os.path.exists(poselift_only_path):
        input_dim = train_poselift[0].shape[1]
        model = TransformerAutoencoder(input_dim, 512, 8, 3)
        model.load_state_dict(torch.load(poselift_only_path))
        model = model.to(device)
        qualitative_testing(model, test_poselift, test_poselift_labels, device, num_samples=10)

    master_model_path = "ClusterTransformerMaster.pth"
    if os.path.exists(master_model_path):
        input_dim = train_poselift[0].shape[1]
        model = TransformerAutoencoder(input_dim, hidden_dims[0], nheads[0], num_layers_list[0])
        model.load_state_dict(torch.load(master_model_path))
        model = model.to(device)
        qualitative_testing(model, test_poselift, test_poselift_labels, device, num_samples=10)

    poselift_model_path = "ClusterTransformerMaster_PoseLift.pth"
    if os.path.exists(poselift_model_path):
        input_dim = train_poselift[0].shape[1]
        model = TransformerAutoencoder(input_dim, hidden_dims[0], nheads[0], num_layers_list[0])
        model.load_state_dict(torch.load(poselift_model_path))
        model = model.to(device)
        qualitative_testing(model, test_poselift, test_poselift_labels, device, num_samples=10)

    

    else:
        if os.path.exists(master_model_path):
            print(f"Found {master_model_path}. Loading for fine-tuning...")

            # Load the model
            input_dim = train_poselift[0].shape[1]
            model = TransformerAutoencoder(input_dim, hidden_dims[0], nheads[0], num_layers_list[0])
            model.load_state_dict(torch.load(master_model_path))
            model = model.to(device)

            print("Starting fine-tuning...")

            model = train_autoencoder_with_eval(
                model,
                train_loader,
                test_loader,  # Use the full test loader for evaluation
                device,
                epochs=100,  # Fine-tune for fewer epochs
                lr=1e-4,  # Use a smaller learning rate for fine-tuning
                save_path="ClusterTransformerMaster_PoseLift.pth"
            )

        else:
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
