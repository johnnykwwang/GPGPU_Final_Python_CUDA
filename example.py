# import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import time

mnist = fetch_mldata("MNIST original")

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

layers = (300,300)
print(layers)

mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=5, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, cuda=True)
t_0 = time.time()
mlp.fit(X_train, y_train)
print("[GPU] Time: %f seconds" % (time.time()-t_0) )
time_gpu = time.time()-t_0
train_score_gpu = mlp.score(X_train, y_train)
test_score_gpu = mlp.score(X_test, y_test)

mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=5, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, cuda=False)
t_0 = time.time()
mlp.fit(X_train, y_train)
time_cpu = time.time()-t_0
train_score_cpu = mlp.score(X_train, y_train)
test_score_cpu = mlp.score(X_test, y_test)
print("[CPU] Time: %f seconds" % (time_cpu) )
print("Time Improvement: %f" % (time_cpu/time_gpu) )
print("Train Score Same = %s " % (train_score_gpu==train_score_cpu) )
print("Test Score Same = %s " % (test_score_gpu==test_score_cpu) )
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
