# import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor
from IPython import embed
import time

boston = load_boston()
X = StandardScaler().fit_transform(boston.data)[: 500]
embed()
X = np.tile(X,(100,1))
y = boston.target[:500]
y = np.tile(y,100)


momentum = 0.5
mlp = MLPRegressor(solver='sgd', max_iter=300, activation='relu',
                           random_state=1, learning_rate_init=0.01,
                           batch_size=X.shape[0], momentum=momentum,cuda=True)
t_0 = time.time()
mlp.fit(X, y)
t_1 = time.time()
pred1 = mlp.predict(X)
score_gpu = mlp.score(X, y)

mlp = MLPRegressor(solver='sgd', max_iter=300, activation='relu',
                           random_state=1, learning_rate_init=0.01,
                           batch_size=X.shape[0], momentum=momentum,cuda=False)
t_2 = time.time()
mlp.fit(X, y)
t_3 = time.time()
print("[GPU] Time: %f seconds" % (t_1-t_0) )
print("[cPU] Time: %f seconds" % (t_3-t_2) )
pred1 = mlp.predict(X)
score_cpu = mlp.score(X, y)
print("[GPU] Score: %f " % (score_gpu) )
print("[cPU] Score: %f " % (score_cpu) )

# print("\x1b[0;33;40mComparison\x1b[0m")
# print("Time Improvement: %f" % (time_cpu/time_gpu) )
# print("Train Score Same ...\t[\x1b[6;30;42m%s\x1b[0m]" % (train_score_gpu==train_score_cpu) )
# print("Test Score Same ...\t[\x1b[6;30;42m%s\x1b[0m]" % (test_score_gpu==test_score_cpu) )
# print("Test Score = %f " % (test_score_gpu) )

# print("Training set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))
# fig, axes = plt.subplots(4, 4)

# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())

# plt.show()
