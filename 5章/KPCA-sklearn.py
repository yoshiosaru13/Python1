from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, random_state=123)

from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

if __name__ == '__main__':
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='r', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='b', marker='o', alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()