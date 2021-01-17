import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import sklearn.datasets

X, y = sklearn.datasets.make_moons(200, noise=0.2)

plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()
import  torch
X = torch.from_numpy(X).type(torch.FloatTensor)
print('y.shape')
print(y.shape)
y = torch.from_numpy(y).type(torch.LongTensor)
# print('X.size(),y.size()')
# print(X.size(),y.size()) torch.Size([200, 2]) torch.Size([200])

import torch.nn as nn
import torch.nn.functional as F


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

model = MyClassifier()
criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #Adam梯度优化器

epochs = 10000
losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred,y)
    print(y_pred.size(),y.size())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

from sklearn.metrics import accuracy_score

print(accuracy_score(model.predict(X), y))

# Output
# 0.995