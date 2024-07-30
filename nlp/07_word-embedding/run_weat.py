import numpy as np
from numpy import dot
from numpy.linalg import norm

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from weat import s, weat_score

target_X = {
    '장미': [4.1, 1.2, -2.4, 0.5, 4.1],
    '튤립': [3.1, 0.5, 3.6, 1.7, 5.8],
    '백합': [2.9, -1.3, 0.4, 1.1, 3.7],
    '데이지': [5.4, 2.5, 4.6, -1.0, 3.6]
}
target_Y = {
    '거미': [-1.5, 0.2, -0.6, -4.6, -5.3],
    '모기': [0.4, 0.7, -1.9, -4.5, -2.9],
    '파리': [0.9, 1.4, -2.3, -3.9, -4.7],
    '메뚜기': [0.7, 0.9, -0.4, -4.1, -3.9]
}
attribute_A = {
    '사랑':[2.8,  4.2, 4.3,  0.3, 5.0],
    '행복':[3.8,  3. , -1.2,  4.4, 4.9],
    '웃음':[3.7, -0.3,  1.2, -2.5, 3.9]
}
attribute_B = {
    '재난': [-0.2, -2.8, -4.7, -4.3, -4.7],
    '고통': [-4.5, -2.1,  -3.8, -3.6, -3.1],
    '증오': [-3.6, -3.3, -3.5,  -3.7, -4.4]
}

X = np.array([v for v in target_X.values()])
Y = np.array([v for v in target_Y.values()])
print(f"X: \n{X}")
print(f"Y: \n{Y}")

A = np.array([v for v in attribute_A.values()])
B = np.array([v for v in attribute_B.values()])
print(f"\nA: \n{A}")
print(f"B: \n{B}")

print('\n장미 - A/B: ', s(target_X['장미'], A, B))
print('거미 - A/B: ', s(target_Y['거미'], A, B))

print('\nX - A/B: ', s(X, A, B))
print('X - A/B (mean): ', round(np.mean(s(X, A, B)), 3))

print('Y - A/B: ', s(Y, A, B))
print('Y - A/B (mean): ', round(np.mean(s(Y, A, B)), 3))
print('X/Y - A/B: ', round(weat_score(X, Y, A, B), 3))

pca = PCA(n_components=2)
pc_A = pca.fit_transform(A)
pc_B = pca.fit_transform(B)
pc_X = pca.fit_transform(X)
pc_Y = pca.fit_transform(Y)

fig, ax = plt.subplots()
ax.scatter(pc_A[:,0],pc_A[:,1], c='blue', label='A')
ax.scatter(pc_B[:,0],pc_B[:,1], c='red', label='B')
ax.scatter(pc_X[:,0],pc_X[:,1], c='skyblue', label='X')
ax.scatter(pc_Y[:,0],pc_Y[:,1], c='pink', label='Y')

plt.savefig("pca.png")
