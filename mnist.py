import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


device = "cuda" if torch.cuda.is_available() else "cpu"

# 前処理
transform = transforms.Compose([
    # 画像をTensorに変換してくれる
    # チャネルラストをチャネルファーストに
    # 0〜255の整数値を0.0〜1.0の浮動少数点に変換してくれる
    transforms.ToTensor()                              
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

num_batches = 100
train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
train_iter = iter(train_dataloader)
# 100個だけミニバッチからデータをロードする
imgs, labels = train_iter.next()

# 100個のデータ, グレースケール, 28px, 28px
imgs.size()

img = imgs[0]
# 画像データを表示するために、チャネルファーストのデータをチャネルラストに変換する
img_permute = img.permute(1, 2, 0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(28 * 28, 400)
        self.l2 = nn.Linear(400, 100)
        self.r2 = nn.ReLU(inplace=True)
        self.l3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.r1(x)
        x = self.l1(x)
        x = self.l2(x)
        feature = x.clone()
        x = self.r2(x)
        x = self.l3(x)
        return x, feature

model = MLP()

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 極端に少ないepoch数で不十分な学習をさせる
num_epochs = 5
losses = []
accs = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.view(num_batches, -1)
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output, f = model(imgs)
        loss = criterion(output, labels)
        running_loss += loss.item()
        # dim=1 => 0-9の分類方向のMax値を返す
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        loss.backward()
        optimizer.step()
    # 600回分で割る
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)
    print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, running_acc))

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

pred_list = []
label_list = []
feature_list = []
for (x, label) in testloader:
    # to GPU
    x = x.view(int(torch.numel(x)/(28*28)), 28*28)
    x = x.to(device)

    output, f = model(x)
    pred = torch.argmax(output, dim=1)
    pred = pred.numpy()
    pred_list.append(pred)
    labels = label.numpy()
    label_list.append(labels)
    f = f.detach().numpy()
    feature_list.append(f)

pred = np.concatenate(pred_list)
labels = np.concatenate(label_list)
features = np.concatenate(feature_list)

print(pred)
print(labels)
print(features)

pca = PCA()
pca.fit(features)
pca_features = pca.transform(features)
print(pca_features)

tsne = TSNE(n_components=2, random_state=0)
tsne_features = tsne.fit_transform(features)
print(tsne_features)


df = pd.DataFrame({
    'index': list(range(pred.size)),
    'label': labels,
    'pred': pred,
    'pca_0': pca_features[:,0],
    'pca_1': pca_features[:,1],
    'tsne_0': tsne_features[:,0],
    'tsne_1': tsne_features[:,1],
})
# len = len(features[0])
# for i in range(len):
#     df[f'feature_{i}'] = features[:,i]
    
df = df.set_index('index')
print(df)

df.to_csv('pred.csv')
