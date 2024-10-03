import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        #self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, max_iter=10, eta=0.01, tolerance=1e-6):
        stoped = False
        weights = []
        weights.append(self.Wout.copy())

        for iteration in range(max_iter):
            num_errors = 0
            for xi, target in zip(X, y):
                pr, hidden = self.predict(xi)
                update = eta * (target - pr)
                self.Wout[1:] += update * hidden.reshape(-1, 1)
                self.Wout[0] += update
                num_errors += int(update != 0.0)
            
            print(f"Итерация {iteration + 1}/{max_iter}, Ошибки: {num_errors}")

            if num_errors == 0:
                stoped = True
                print("Обучение сошлось.")
                break

            current_weight = self.Wout.copy()
            if (any(sum(abs(current_weight - weight)) < tolerance for weight in weights)):
                stoped = True
                print("Изменение весов ниже порога точности. Алгоритм зациклился.")
                break
            
            weights.append(current_weight)

        if not stoped:
            print("Достигнуто максимальное количество итераций.")

        return self

