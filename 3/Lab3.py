import numpy as np
import pandas as pd
import csv
import random
from datetime import date
import matplotlib.pyplot as plt

def table():
    with open('Muzh.txt', 'r', encoding='utf-8') as f:
        Muzh = [i.rstrip() for i in f]

    with open('Zhen.txt', 'r', encoding='utf-8') as f:
        Zhun = [i.rstrip() for i in f]

    FootballLists = {
        "ЦСК": ["Нападающий", "Защитник",
                                "Полузащитник", "Вратарь"],
        "Зенит": ["Нападающий", "Защитник",
                                "Полузащитник", "Вратарь"],
        "Спартак": ["Нападающий", "Защитник",
                                "Полузащитник", "Вратарь"],
    }
    # Заполнение CSV
    with open("Football_list.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                "Табельный номер", "Фамилия.И.О.", "Пол", "Год рождения", "Год начала работы",
                "Клуб", "Позиция", "Зарплата", "Колличество голов"
            )
        )
    # Генерация Имени и Отчества
    def Imya_Och():
        return (random.choice('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ') + ". " + random.choice(
            'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ') + ".")
    # Генерация таблицы футболистов
    for Players in range(random.randint(1000, 1111)):
        Football = random.choice(list(FootballLists))
        sex = random.choice(["Мужчина", "Женщина"])
        Birthday = random.randint(1950, 2003)
        StartingYear = min(date.today().year, Birthday + random.randint(19, 99))
        Salary = random.randrange(10000, 10000000, 10000)
        Goals = random.randint(0, 100)

        if (sex == "Мужчина"):
            Name = Muzh[random.randint(0, len(Muzh) - 1)]
        else:
            Name = Zhun[random.randint(0, 111)]
        # Заполнение CSV
        with open("Football_list.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    Players + 1,
                    (Name + " " + Imya_Och()), sex, Birthday, StartingYear,
                    Football, random.choice(FootballLists[Football]),
                    Salary, Goals
                ]
            )
# Работа с помощью NUMPY
def numpy():
    with open("Football_list.csv") as csvfile:
        metadata = [list(row) for row in csv.reader(csvfile)]
    data = np.array(metadata)
    sexs = data[:, 2]
    Football = data[:, 5]
    Salary = data[1:, 7].astype("int32")
    StartingYear = data[1:, 4].astype("int32")
    count_Salary = np.bincount(Salary)
    count_StartingYear = np.bincount(StartingYear)
    # Вывод данных в консоль(Numpy)
    print("Задание по NUMPY \n ---------------- \n Статистика по футболистам \n "
          "Количество футболистов всего:", np.count_nonzero(Salary), "\nПодъём зарплат был в",
          np.argmax(count_StartingYear), "году. \nТак-как колличество новых футболистов больше в этом году"
          "\n----------------" "\nДоля футболистов мужского пола:", round(np.sum(sexs == "Мужчина") / np.size(sexs), 3),
          "\nДоля футболистов женского пола:", round(np.sum(sexs == "Женщина") / np.size(sexs), 3), "\n----------------"
          "\nЗарплаты футболистов \n Максимальная зарплата футболистов:", np.max(Salary),
          "\nМинимальная зарплата футболистов:", np.min(Salary), "\nСреднея зарплата футболистов:",
          round(np.average(Salary), 3), "\nДисперсия заплат футболистов",
          round(np.var(Salary), 3), "\nСт. откл. зарплат футболистов:", round(np.std(Salary), 3),
          "\nМедиана зарплат футболистов:",np.median(Salary), "\nМода зарплат футболистов:",
          np.argmax(count_Salary), "\n----------------\nГруппировка футболистов по клубам"
                                           "\nКоличество футболистов ЦСК",
          np.count_nonzero(Football == "ЦСК"), "\nКоличество футболистов Зенит:",
          np.count_nonzero(Football == "Зенит"), "\nКоличество футболистов Спартак:",
          np.count_nonzero(Football == "Спартак"), "\nБольше всего сотрудников в:",
          "ЦСК" if ((np.count_nonzero(Football == "ЦСК") \
                                            > np.count_nonzero(Football == "Зенит")) and (
                                                       np.count_nonzero(Football == "ЦСК") \
                                                       > np.count_nonzero(Football == "Спартак")))
          else ("Зените" if (np.count_nonzero(Football == "Зенит") \
                                         > np.count_nonzero(Football == "Спартак"))
                else "Спартаке"), "\n----------------")


# Работа с помощью PANDAS
def pandas():
    data = pd.read_csv("Football_list.csv", encoding='cp1251')
    # Вывод данных в консоль(Pandas)
    print("Задание PANDAS\n----------------\nСтатистика футболистов\nОбщее колличество футболистов:", data["Табельный номер"].count(),
          "\nДоля футболистов мужского пола:", round(data["Пол"].value_counts()["Мужчина"] / data["Пол"].shape[0], 3),
          "\nДоля футболистов женского пола:", round(data["Пол"].value_counts()["Женщина"] / data["Пол"].shape[0], 3),
          "\nБольше всего футболистов появилось в", data["Год начала работы"].mode()[0], "году\n----------------"
          "\nЗарплата\nМаксимальное зарплата футболистов:", data["Зарплата"].max(),
          "\nМинимальное зарплата футболистов:", data["Зарплата"].min(), "\nДисперсия зарплаты:",
          round(data["Зарплата"].var(), 3), "\nСт. откл. зарплаты олгархов:", round(data["Зарплата"].std(), 3),
          "\nМедиана кол-ва зарплаты футболистов:", data["Зарплата"].median(), "\nМода зарплаты",
          data["Зарплата"].mode()[0], "\n----------------\nГруппировка футболистов"
                                       "\nКоличество футболистов в ЦСК:",
          (data["Клуб"].value_counts()["ЦСК"]),
          "\nКоличество футболистов в Зенит:",
          (data["Клуб"].value_counts()["Зенит"]),
          "\nКоличество футболистов в Спартак:",
          (data["Клуб"].value_counts()["Спартак"]),
          "\nБольше всего футболистов в:",
          "ЦСК" if (
                      (data["Клуб"].value_counts()["ЦСК"] \
                       > data["Клуб"].value_counts()["Зенит"]) and (
                              data["Клуб"].value_counts()[
                                  "ЦСК"] \
                              > data["Клуб"].value_counts()[
                                  "Зенит"]))
          else ("Зените" if (data["Клуб"].value_counts()
                                           ["Зенит"] \
                          > data["Клуб"].value_counts()[
                              "Спартак"]) else "Спартаке"))
# Работа с графиками(3шт)
def graphics():
    data = pd.read_csv("Football_list.csv", encoding='cp1251')
    proportion = {}
    positions = data['Позиция'].unique()

    for item in positions:
        pos = data[data['Позиция'] == item]
        proportion[item] = round(pos['Колличество голов'].sum() / pos['Колличество голов'].count(), 2)
    # График распределения соотношения по полу(мужской или женский)
    sex = [data["Пол"].value_counts()["Мужчина"], data["Пол"].value_counts()["Женщина"]]
    plt.pie(sex, labels=["Мужчины", "Женщины"])
    plt.title('Соотношение футболистов по полу', loc='center')
    plt.show()
    # График группировки по области нахождения(ЦСК, Спартак, Зенит)
    plt.hist(data['Клуб'], bins=8, color='y')
    plt.title('Группировка футболистов', loc='center')
    plt.show()
    # График по колличеству футболистов в обределённой должности(миллионер или миллиардер, рублейвой или доллоровый)
    plt.bar(proportion.keys(), proportion.values(), color='r')
    plt.title('Соотношение количества предприятий по должностям', loc='right')
    plt.show()

table()
numpy()
pandas()
graphics()
