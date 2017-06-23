# import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from IPython import embed
import time

mnist = fetch_mldata("MNIST original")

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

layers = (300,100)
print(layers)

# mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=5, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1, cuda=True)
# t_0 = time.time()
# mlp.fit(X_train, y_train)
# print("[GPU] Time: %f seconds" % (time.time()-t_0) )
# time_gpu = time.time()-t_0
# train_score_gpu = mlp.score(X_train, y_train)
# test_score_gpu = mlp.score(X_test, y_test)

batch_sizes = [1000,1500,2000,2500,3000]
iterations = [10,15,20,25,30]
results = []
for bs in batch_sizes:
    for it in iterations:
        mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=it, alpha=1e-4,
                    solver='sgd',batch_size=bs, verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, cuda=False)
        t_0 = time.time()
        mlp.fit(X_train, y_train)
        time_cpu = time.time()-t_0
        test_score_cpu = mlp.score(X_test, y_test)
        results.append({'batch_size':bs,'max_iter':it,'time':time_cpu,'score':test_score_cpu})
        print("[CPU] Time: %f seconds" % (time_cpu) )
        print("[CPU] Score = %f" % (test_score_cpu) )

f = open("results/classifier_cpu.txt","a")
# f.write("batch_size,max_iter,time,score\n")
for r in results:
    f.write( "%d,%d,%f,%f\n" %(r['batch_size'],r['max_iter'],r['time'],r['score']) )

results = []
for bs in batch_sizes:
    for it in iterations:
        mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=it, alpha=1e-4,
                    solver='sgd',batch_size=bs, verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, cuda=True)
        t_0 = time.time()
        mlp.fit(X_train, y_train)
        time_gpu = time.time()-t_0
        test_score_gpu = mlp.score(X_test, y_test)
        results.append({'batch_size':bs,'max_iter':it,'time':time_gpu,'score':test_score_gpu})
        print("[GPU] Time: %f seconds" % (time_gpu) )
        print("[GPU] Score = %f" % (test_score_gpu) )

f = open("results/classifier_gpu.txt","a")
# f.write("batch_size,max_iter,time,score\n")
for r in results:
    f.write( "%d,%d,%f,%f\n" %(r['batch_size'],r['max_iter'],r['time'],r['score']) )
# print("Time Improvement: %f" % (time_cpu/time_gpu) )
# print("Train Score Same = %s " % (train_score_gpu==train_score_cpu) )
# print("Test Score Same = %s " % (test_score_gpu==test_score_cpu) )
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
