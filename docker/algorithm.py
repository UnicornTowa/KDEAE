import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KernelDensity


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=2),  # 11
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, stride=2),  # 5
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=2),  # 2
            nn.Flatten(),
            nn.Linear(16 * 2, 4)

        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16 * 2),
            nn.Unflatten(1, (16, 2)),
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def kdeae(data: np.ndarray, args) -> np.ndarray:
    print(args)
    print(type(args))
    print('Preparing...')
    seq_len = 23
    skip = len(data) % seq_len
    data = data.astype(np.float32)
    data = data[:len(data) - skip]
    data = (data - data.mean()) / (data.std() + (data.std() == 0).astype('float'))
    data = data.reshape(-1, seq_len)

    all_data = torch.tensor(data, dtype=torch.float32)
    batch_size = 32
    all_loader = DataLoader(TensorDataset(all_data), batch_size=batch_size, shuffle=True)

    model = ConvAutoencoder(bottleneck_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    print('Training...')
    epochs = args.customParameters.epochs
    for _ in range(epochs):
        model.train()
        for batch in all_loader:
            batch = batch[0].unsqueeze(1).to(device)
            preds = model(batch)
            loss = criterion(preds, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('Predicting...')
    encoded = []
    with torch.no_grad():
        for sample in all_data:
            inputs = sample.view(1, 1, -1).to(device)
            features = model.encoder(inputs).view(-1)
            encoded.append(features.cpu().detach().numpy())

    encoded = np.array(encoded)
    encoded_norm = (encoded - encoded.mean(axis=0)) / encoded.std(axis=0)

    print('Scoring...')

    medians = np.median(encoded_norm, axis=0)
    middle = np.repeat(np.inf, 4)
    cur_min = np.inf
    for sample in encoded_norm:
        dist = np.dot(sample - medians, sample - medians)
        if dist < cur_min:
            cur_min = dist
            middle = sample
    q = args.customParameters.band_param
    dists = np.linalg.norm(encoded_norm - middle, axis=1)
    band = max(np.quantile(dists, q), 1e-3)

    kde = KernelDensity(kernel='gaussian', bandwidth=band)
    kde.fit(encoded_norm)
    kde_scores = kde.score_samples(encoded_norm)
    q3 = np.quantile(kde_scores, 0.75)
    q1 = np.quantile(kde_scores, 0.25)
    kde_scores = ((kde_scores - np.median(kde_scores)) /
                  (q3 - q1 + (1 if q3 == q1 else 0)))
    p = args.customParameters.sigmoid_pow

    kde_scores = 1 / (1 + np.exp(p * kde_scores))

    pred_proba = np.array([[kde_scores[i]] * seq_len for i in range(len(kde_scores))]).flatten()
    anomaly_score = np.zeros(len(data) * 23 + skip)
    anomaly_score[:seq_len * len(kde_scores)] = pred_proba

    return anomaly_score


@dataclass
class CustomParameters:
    epochs: int = 200
    band_param: float = 0.004
    sigmoid_pow: float = 1


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(args):
    df = pd.read_csv(args.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    return data, labels


def run(args):
    x, _ = load_data(args)
    scores = kdeae(x, args)
    np.savetxt(args.dataOutput, scores, delimiter=",")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    run(args)
