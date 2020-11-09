import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

digits = load_digits()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.2, random_state=2)
clf = svm.SVC(gamma=0.001, C=100., probability=True)
clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest) # 输出可能性最高
print(clf.predict_proba(Xtest[1:4])) # 输出各种可能性的概率
print(clf.score(Xtest, Ytest))
print(accuracy_score(Ytest, Ypred))

# 画图
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap=plt.cm.gray_r)
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32,
            transform=ax.transAxes, color='green' if Ypred[i] == Ytest[i] else 'red')
    ax.text(0.8, 0.05, str(Ytest[i]), color='black'
            , fontsize=32, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
# plt.show()
joblib.dump(clf, 'digits_svm.pk1')
clf = joblib.load('digits_svm.pk1')
Ypred = clf.predict(Xtest)
print(accuracy_score(Ytest, Ypred))
