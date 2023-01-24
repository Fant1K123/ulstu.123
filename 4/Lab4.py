import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pylab
# Задание класcификаций для продуктов
Parametrs = ['Продукт', 'Сладость', 'Хруст', 'Класс']
# Создание списка с продуктами и их параметрами
Food_1 = [['Яблоко', '7', '7', '0'],
        ['Салат', '2', '5', '1'],
        ['Бекон', '1', '2', '2'],
        ['Банан', '9', '1', '0'],
        ['Орехи', '1', '5', '2'],
        ['Рыба', '1', '1', '2'],
        ['Сыр', '1', '1', '2'],
        ['Виноград', '8', '1', '0'],
        ['Морковь', '2', '8', '1'],
        ['Апельсин', '6', '1', '0'],
        ['Мандарин', '9', '1', '0'],
        ['Арбуз', '3', '7', '1'],
        ['Тыква', '3', '1', '2'],
        ['Репка', '5', '3', '0'],
        ['Подсолнух', '1', '7', '1']]
#Создание разметки
def Setka(x1, y1, x2, y2):
    return (abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2) ** 0.5
# Реализация классификатора по методу k-NN
def kNN_CLASS(products, OriginalSIZE, pars_window, Res):
# Индексация переменных
    Materials = np.array(products)
    Size1 = len(Materials) - OriginalSIZE
    Dist = np.zeros((Size1, OriginalSIZE))
    All_class = [0] * Size1
# Реализация классификатора №1
    for i in range(Size1):
        for j in range(OriginalSIZE):
            Distanse = Setka(int(Materials[OriginalSIZE + i][1]),
                            int(Materials[OriginalSIZE + i][2]), int(Materials[j + 1][1]),
                            int(Materials[j + 1][2]))
            Dist[i][j] = Distanse if Distanse < pars_window else 1000
# Реализация классификатора №2
    for i in range(Size1):
        print(str(i) + ') ' + Materials[OriginalSIZE + i][0])
        Massa = [0] * products.iloc[:]['Класс'].nunique()
        neighbor = np.sum(Dist[i] != 1000)
        for j in range(neighbor + 1):
            min = Dist[i].argmin()
            Massa[int(Materials[min + 1][3])] += ((neighbor - j + 1) / neighbor)
            Dist[i][min] = 1000
        All_class[i] = np.array(Massa).argmax()
# Проверка классов (Предположительный и Оригинальный)
# !Должны все совпадать!
# Сообщает о правильности проведения эксперемента!
        print('Предположительный класс: ', All_class[i], 'Оригинальный класс: ', Materials[OriginalSIZE + i][3])
        if int(All_class[i]) != int(Materials[OriginalSIZE + i][3]):
            print('нет совпадения')
        else:
            print('совпадение')
            Res += 1
    print(All_class)
    print('Совпадений ручного классификатора: ', str(Res))
    return All_class
# Реализация классификатора методом k-NN библиотеки sklearn
def sklearnKNN(values, k, y):
    x_train, x_test, y_train, y_test = train_test_split(values, y, test_size=0.3, shuffle=False, stratify=None)
    scal = StandardScaler()
    scal.fit(x_train)
    x_train = scal.transform(x_train)
    x_test = scal.transform(x_test)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    return x_train, x_test, y_train, y_test, predict
def Demonstration(data, window, colors):
    x = data['Сладость']
    y = data['Хруст']
    Colors_View = [colors[str(i)] for i in data['Класс']]
    pylab.subplot(2, 1, window)
    plt.scatter(x, y, c=Colors_View)
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')
# запись данных в файл
with open('data1.csv', 'w', encoding='utf8') as f1:
    writer = csv.writer(f1, lineterminator="\r")
    writer.writerow(Parametrs)
    for row in Food_1:
        writer.writerow(row)
# чтение данных из файла
Materials = pd.read_csv('data1.csv')
train = 10
k = 4
X_for_knn1 = Materials.iloc[:, 1:3].values
Y_for_knn1 = Materials.iloc[:, 3].values
All_class = Materials[:train]['Класс']
# Использование KNN для данных номер 1!
knn_res = pd.Series(kNN_CLASS(Materials, train, 4, 0))
All_class = pd.concat([All_class, knn_res])
color_knn_view = {'0': 'green', '1': 'black', '2': 'brown'}
Demonstration(Materials, 1, color_knn_view)
colors_spinner_view = [color_knn_view[str(i)] for i in All_class]
plt.show()
# ИСпользование knn1 с библиотекой sklearn
X_train, X_test, y_train, y_test, predict = sklearnKNN(X_for_knn1, k, Y_for_knn1)
# вывод статистики качества прогнозирования
print('Cтатистика качества прогнозирования\n', classification_report(y_test, predict),
      confusion_matrix(y_test, predict))
Food_2 = [['Яблоко', '7', '7', '0'],
            ['Салат', '2', '5', '1'],
            ['Арбуз', '10', '8', '3'],
            ['Яйца', '1', '2', '2'],
            ['Банан', '9', '1', '0'],
            ['Фундук', '10', '9', '3'],
            ['Орехи', '1', '5', '2'],
            ['Рыба', '1', '1', '2'],
            ['Сыр', '1', '1', '2'],
            ['Виноград', '8', '1', '0'],
            ['Шипучка', '8', '10', '3'],
            ['Морковь', '2', '8', '1'],
            ['Апельсин', '6', '1', '0'],
            ['конфета', '6', '9', '3'],
            ['Малина', '9', '1', '0'],
            ['Капуста', '3', '7', '1'],
            ['Халва', '6', '7', '3'],
            ['Говядина', '3', '1', '2'],
            ['Свиника', '5', '3', '0'],
            ['Свекла', '1', '7', '1'],
            ['Мармелад', '7', '8', '3']]
# запись новых данных во второй файл
with open('data2.csv', 'w', encoding='utf8') as f2:
    writer = csv.writer(f2, lineterminator="\r")
    writer.writerow(Parametrs)
    for row in Food_2:
        writer.writerow(row)
# загрузка новых данных
data2 = pd.read_csv('data2.csv')
train = 14
# Использование KNN для данных номер 2!
knn_res = pd.Series(kNN_CLASS(data2, train, k, 0))
data1 = pd.concat([All_class, knn_res])
X_for_knn2 = data2.iloc[:, 1:3].values
Y_for_knn2 = data2.iloc[:, 3].values
# Использование knn2 с библиотекой sklearn
X_train, X_test, y_train, y_test, predict = sklearnKNN(X_for_knn2, k, Y_for_knn2)
# вывод статистики качества прогнозирования
print('Cтатистика качества прогнозирования\n', classification_report(y_test, predict),
      confusion_matrix(y_test, predict))
color_knn_view = {'0': 'green', '1': 'black', '2': 'brown', '3': 'yellow'}
Demonstration(data2, 1, color_knn_view)
plt.show()