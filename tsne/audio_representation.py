import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import os


def save_plot(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig = plt.gcf()

    fig.savefig(save_path, dpi=300)
    print('png saved in: ', save_path)

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright",2)

modal = 'audio'
work_dir = '/m_fusion_data/'
path =  work_dir + f'representation/{modal}.npz'
print(path)
data = np.load(path)
print(data.files)
class_name=['non-sarcastic','sarcastic']
X = data['repre']
y4 = data['label']
y4 = [class_name[yi] for yi in y4]
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
# print(y4)
print(X_embedded.shape)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y4, legend='full', palette=palette)
plt.title(modal,fontsize=20)
plt.legend(fontsize=20)
# plt.show()
path =  work_dir + f'representation/img/{modal}.png'
save_plot(path)
