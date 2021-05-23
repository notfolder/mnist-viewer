import pandas as pd
from torchvision import datasets
from PIL import Image
from sklearn.manifold import TSNE

#df = pd.read_csv('pred.csv')
#print(df.groupby(['label','pred']))

# testset = datasets.MNIST(root='./data', train=False, download=True)
# im = testset.data[0].detach().numpy()
# print(im)
# img = Image.fromarray(im)
# img.show()

df = pd.read_csv('pred.csv')
tsne = TSNE(n_components=2, random_state=0)
feature_list = []
for i in range(100):
    feature_list.append(f'feature_{i}')
feature = tsne.fit_transform(df[feature_list])
