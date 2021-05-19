import torch
from torch import nn
from torch import Tensor
#from torch._C import dtype, float32
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from data.utils import load_data, split_data
from data.scaling import normalize

import typing
import math


# from layers.time2vec import Time2Vector

# from tensorflow.keras import layers
# import tensorflow as tf

import numpy as np

coin = 'BTC'
df = load_data(coin, 'bitcoin').iloc[:, :]

# df_floats = df.select_dtypes(include=float).astype('float32')
# df_not_float = df.select_dtypes(exclude=float)
# df = df_floats.join(df_not_float)

used_features_names = ['prices', 'variation', 'trends', 'events']


class CryptoCurrencyDataset(Dataset):
    def __init__(self, dataframe, n_past_days=128, foward_days=1, dtype=np.float32):
        df = dataframe
        N = n_past_days

        prices = normalize(df.loc[:, 'price'].to_numpy())[0]
        variation = df.loc[:, 'variation (%)'].to_numpy()
        tweet = normalize(df.loc[:, 'tweet_volume'].to_numpy())[0]
        google_trends = normalize(df.loc[:, 'trend'].to_numpy())[0]
        event_day_count = df.loc[:, 'days_to_event_happen'].fillna(0).to_numpy()
        event_votes = normalize(df.loc[:, 'event_votes'].fillna(0).to_numpy())[0]
        event_confidence = normalize(df.loc[:, 'event_confidence'].fillna(0).to_numpy())[0]

        #features = [prices, variation, google_trends, event_day_count, event_votes, event_confidence]
        features = [prices]

        n_samples = len(prices) - N - (foward_days-1)

        # matriz de dimensao: [n_samples, N, n_features];
        self.features_set = np.empty((n_samples, N, len(features)), dtype=dtype)  # type: np.ndarray
        self.labels = [None]*n_samples
        for i in range(n_samples):
            for j in range(N):
                for ft_idx in range(len(features)):
                    feature = features[ft_idx]
                    self.features_set[i, j, ft_idx] = feature[i+j]

            self.labels[i] = prices[N+i: N+i+foward_days]
        self.labels = np.array(self.labels, dtype=dtype)

    @property
    def shape(self) -> typing.Any:
        'Tuple of array dimensions.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3, 4])\n    >>> x.shape\n    (4,)\n    >>> y = np.zeros((2, 3, 4))\n    >>> y.shape\n    (2, 3, 4)'
        return self.features_set.shape

    def __len__(self):
        return len(self.features_set)

    def __getitem__(self, idx):
        # sample = {'target': torch.from_numpy(self.labels[idx]),
        #           'features': torch.from_numpy(self.features_set[idx])}
        return self.labels[idx], self.features_set[idx]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        return self.dropout(x + self.pe[:x.size(0), :])

class Transformer(nn.Module):

    def __init__(self, n_features, n_head, num_layers=3, dropout=0.1):
        super(Transformer, self).__init__()

        self.pos_encoder = PositionalEncoding(n_features, dropout=0.)
        # d_model: numero de features
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(n_features, 1)  # todo: será se ta certo isso daqui?? acho que nao em

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)    # todo: confuso, investigar
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(.0))
        return mask

    def forward(self, src, device):
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return output


device = torch.device('cuda')
train, test = split_data(df, ratio=0.9)

# features = np.array(features)
# features = torch.from_numpy(features)

# model = Transformer(n_features=features.shape[0], n_head=3).to(device)
# optimizer = torch.optim.Adam(model.parameters())

# mse_criterion = torch.nn.MSELoss()


# model.train()

# batch_size = 1
# data = torch.from_numpy(features_set[0:batch_size, :, :]).to(device)

# # Shape of features_set : [batch, input_length, feature] (a.k.a [samples, input_length, feature])
# # Desired input for model: [input_length, batch, feature]
# data = data.permute(1, 0, 2).to(device)
# target = torch.from_numpy(labels[0]).to(device)
# prediction = model(data, device)
# loss = mse_criterion(prediction, target)

# loss.backward()
# optimizer.step()


train_dataset = CryptoCurrencyDataset(train)
train_dataLoader = DataLoader(train_dataset, batch_size=1, shuffle=False)

model = Transformer(n_features=train_dataset.shape[2], n_head=1).to(device)
optimizer = torch.optim.Adam(model.parameters())

mse_criterion = torch.nn.MSELoss()
epochs = 100


for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    # teacher forcing training method
    model.train()
    for target, features in train_dataLoader:
        optimizer.zero_grad()

        # Shape of features_set : [batch, input_length, feature] (a.k.a [samples, input_length, feature])
        # Desired input for model: [input_length, batch, feature]
        data = features.permute(1, 0, 2).to(device)
        target = target.to(device)
        target2 = torch.cat([data[1:], target.reshape(1,-1,1)]).to(device)
        prediction = model(data, device)
        loss = mse_criterion(prediction, target2)

        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

    if epoch % 10 == 0: #plot 1-step precitions
        import matplotlib.pyplot as plt

        target_plot = target2[:,:,0].detach().cpu().numpy()
        pred_plot = prediction[:,:,0].detach().cpu().numpy()

        idx_target = [i for i in range(len(target_plot))]
        idx_pred = [i for i in range(1, len(pred_plot)+1)]

        plt.figure(figsize=(15,6))
        plt.rcParams.update({"font.size" : 18})
        plt.grid(b=True, which='major', linestyle = '-')
        plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
        plt.minorticks_on()

        plt.plot(idx_target, target_plot, 'o-.', color = 'blue', label = 'target sequence', linewidth=1)
        plt.plot(idx_pred, pred_plot, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)

        plt.title("Teaching Forcing, Epoch " + str(epoch))
        plt.xlabel("Time Elapsed")
        plt.ylabel("price")
        plt.legend()
        path_to_save = 'graph'
        plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
        plt.close()


    train_loss /= len(train_dataLoader)

    print(f'Epoch {epoch}/{epochs} - loss = {train_loss}')


test_dataset = CryptoCurrencyDataset(test)
test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []

for target, features in test_dataLoader:
    data = features.permute(1, 0, 2).to(device)
    target = target.to(device)
    prediction = model(data, device)
    loss = mse_criterion(prediction, target)

    predictions.append(prediction)



############ DISPLAYING DATA ############
from data.visualization import plot_data
prices_test = test['price'].to_list()
# prediction tem um shape (samples, foward_days), entao vamos pegar previsoes de foward_days dias, a cada partir de pontos com intervalo de foward_days
foward_days_predictions = []
for i in range(1, len(predictions)):
    foward_days_predictions.append(np.array([None]*(i-1) + [prices_test[i-1]] +
                                            predictions[i].tolist() +
                                            [None]*(len(predictions)-(i+1))))




fig = plot_data([prices_test] + foward_days_predictions, tick=10, legends=['actual_price', 'prediction'],
                colors=['b', 'r'] + ['r']*len(foward_days_predictions), blocking=False)

# 1 day predictions
fig = plot_data([prices_test, predictions[:, 0]], tick=10, legends=['actual_price', 'prediction'], blocking=False)

# l = Time2Vector(N, linear_used_features_shape=(1,))

# input_shape = (32, N, len(features)) #(batch_size, seq_len, total_features)   seq_len == N
# l.build(input_shape)
# l.call(features_set)

# num_heads = 2
# key_dim = len(features)

# multiAttention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
# multiAttention(features_set, features_set, return_attention_scores=True)
