# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from Staroverov_neural import MLP
from sklearn.preprocessing import LabelBinarizer


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]

y = df.iloc[0:100, 4].values
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

X = df.iloc[0:100, [0, 1, 2, 3]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = y.shape[1] # количество выходных сигналов равно количеству классов задачи

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):
    net.train_stochastic(X, y)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - net.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net.predict(X)
predicted_classes = np.argmax(pr, axis=1)
true_classes = np.argmax(y, axis=1)
num_incorrect = np.sum(predicted_classes != true_classes)
print("Количество неверных ответов:", num_incorrect)