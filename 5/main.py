import matplotlib.pyplot as pyplot
import pandas
import pylab
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


df = pandas.read_csv("titanic.csv")
teach_df = df.iloc[:100]
test_df = df.iloc[100:]


def init(df):
    df = df.copy()
    df = df.drop("Name", axis=1)
    df = df.drop("Age", axis=1)
    df = df.drop("Fare", axis=1)
    df = df.drop("Siblings/Spouses Aboard", axis=1)
    df = df.drop("Parents/Children Aboard", axis=1)
    df['Sex'] = pandas.factorize(df['Sex'])[0]
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X = pandas.DataFrame(X, index=X.index, columns=X.columns)
    return X, y


scal = StandardScaler()
X_teach, y_teach = init(teach_df)
X_teach = scal.fit_transform(X_teach)
X_test, y_test = init(test_df)
X_test = scal.fit_transform(X_test)

knn = KNeighborsClassifier().fit(X_teach, y_teach)
knn_predict = pandas.Series(knn.predict(X_test))
print("Точность предсказаний KNN: " + str(knn.score(X_test, y_test) * 100) + "%")

lda = LinearDiscriminantAnalysis().fit(X_teach, y_teach)
lda_predict = pandas.Series(lda.predict(X_test))
print("Точность предсказаний методом LDA: " + str(lda.score(X_test, y_test) * 100) + "%")


svm = SVC(kernel="rbf").fit(X_teach, y_teach)
svm_predict = pandas.Series(svm.predict(X_test))
print("Точность предсказаний SVM: " + str(svm.score(X_test, y_test) * 100) + "%")

pylab.figure(figsize=(20, 10))
pylab.subplot(1, 2, 1)
pyplot.pie(y_test.value_counts().sort_index(), labels=sorted(y_test.unique()), autopct="%1.1f%%")
pyplot.title("Текущая оценка вероятности")
pylab.subplot(1, 2, 2)
pyplot.pie(knn_predict.value_counts().sort_index(), labels=sorted(knn_predict.unique()), autopct="%1.1f%%")
pyplot.title("Оценка вероятности по KNN")
pyplot.show()

pylab.figure(figsize=(20, 10))
pylab.subplot(1, 2, 1)
pyplot.pie(y_test.value_counts().sort_index(), labels=sorted(y_test.unique()), autopct="%1.1f%%")
pyplot.title("Выжил \ не выжил")
pylab.subplot(1, 2, 2)
pyplot.pie(lda_predict.value_counts().sort_index(), labels=sorted(lda_predict.unique()), autopct="%1.1f%%")
pyplot.title("Выжил \ не выжил по LDA")
pyplot.show()

pylab.figure(figsize=(20, 10))
pylab.subplot(1, 2, 1)
pyplot.pie(y_test.value_counts().sort_index(), labels=sorted(y_test.unique()), autopct="%1.1f%%")
pyplot.title("Текущая оценка вероятности")
pylab.subplot(1, 2, 2)
pyplot.pie(svm_predict.value_counts().sort_index(), labels=sorted(svm_predict.unique()), autopct="%1.1f%%")
pyplot.title("Оценка вероятности по SVM")
pyplot.show()
