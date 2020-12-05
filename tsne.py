from sklearn.manifold import TSNE
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser(description='AutoGRL')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--model', type=str, help='model name')
args = parser.parse_args()
z, y = pickle.load(open(f'embeddings/{args.dataset}_{args.model}.pt', 'rb'))

z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
y = y.cpu().numpy()
plt.figure(figsize=(8, 8))
num_classes = int(max(y)) + 1

colors = list(mcolors.CSS4_COLORS.values())

for i in range(num_classes):
    plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i + 10])

plt.axis('off')
plt.savefig(f'figures/{args.dataset}_{args.model}.jpg')

