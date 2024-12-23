import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Veri seti yükleme ve bölme
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

# Veriyi ölçekleme
scaler = StandardScaler()
train_scaled = scaler.fit_transform(x_train)
valid_scaled = scaler.transform(x_valid)

# Tensor dönüşümleri
x_train = torch.from_numpy(train_scaled.astype("float32"))
x_valid = torch.from_numpy(valid_scaled.astype("float32"))
y_train = torch.from_numpy(y_train.astype("float32")).view(-1, 1)
y_valid = torch.from_numpy(y_valid.astype("float32")).view(-1, 1)

# Cihaz kontrolü (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Verileri cihaza taşı
x_train = x_train.to(device)
x_valid = x_valid.to(device)
y_train = y_train.to(device)
y_valid = y_valid.to(device)

# Eğitim parametreleri
lr = 0.001  # Daha uygun bir öğrenme oranı
epoch = 100
n_samples, n_features = x_train.shape

# Model sınıfı
class LogisticRegressor(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
    def fit(self, x, y):
        # Loss ve optimizer
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr)
        
        # Modeli cihaza taşı
        self.to(device)
        
        for epoch_idx in range(epoch):
            # İleri yayılım
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            
            # Geri yayılım
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # İlerlemeyi yazdır
            if epoch_idx % 2 == 0:
                [w, b] = self.linear.parameters()
                print(f"Epoch {epoch_idx + 1}: Weight = {w[0][0].item()}, Loss = {loss.item()}")
    
    def predict(self, x):
        with torch.no_grad():
            y_predicted = self.forward(x)
            return y_predicted.round()

# Modeli eğit
model = LogisticRegressor(n_features)
model.fit(x_train, y_train)
y_predicted_cls=model.predict(x_valid)


with torch.no_grad():
   
    acc = y_predicted_cls.eq(y_valid).sum() / float(y_valid.shape[0])
    print(f'accuracy: {acc.item():.4f}')