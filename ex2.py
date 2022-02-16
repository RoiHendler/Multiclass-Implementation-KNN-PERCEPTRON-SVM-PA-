import numpy as np
import sys


def shuffle_xy(X, y):
    '''
    Shuffle X and y in the same random order.
    '''
    indices = np.arange(y.size)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

def test(model, X, y):
    '''
    Test the model on the given examples and get the correct precentage.
    '''
    correct = 0
    for x, y1 in zip(X, y):
        if model.predict(x) == y1:
            correct += 1
    return correct / y.size


class KNN:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, t):
        distances = np.array([np.linalg.norm(x - t) for x in self.X])
        p = distances.argsort()
        best_k = self.y[p][:self.k]
        return np.argmax(np.bincount(best_k))


class Perceptron:
    def __init__(self, X, y, eta=1, epochs=500):
        self.X = X
        self.y = y

        w = np.zeros((3, X[0].size))

        for _ in range(epochs):
            X, y = shuffle_xy(self.X, self.y)

            for x1, y1 in zip(X, y):
                y_hat = np.argmax(np.dot(w, x1))
                if y1 != y_hat:
                    w[y1, :] = w[y1, :] + eta * x1
                    w[y_hat, :] = w[y_hat, :] - eta * x1

            eta *= 0.9

        self.w = w

    def predict(self, t):
        return np.argmax(np.dot(self.w, t))


class SVM:
    def __init__(self, X, y, eta=1, lamda = 0.01, epochs=1000):
        self.X = X  # np.hstack((np.ones((X.shape[0], 1)), np.delete(X, 4, axis=1)))
        self.y = y

        w = np.zeros((3, X[0].size))

        for _ in range(epochs):
            X, y = shuffle_xy(self.X, self.y)

            for x1, y1 in zip(X, y):
                w *= (1 - lamda * eta)
                y_hat = np.argmax(np.dot(w, x1))
                if y1 != y_hat:
                    w[y1, :] = w[y1, :] + eta * x1
                    w[y_hat, :] = w[y_hat, :] - eta * x1

            eta *= 0.9

        self.w = w

    def predict(self, t):
        # t = np.insert(t, 0, 1)[:-1]
        return np.argmax(np.dot(self.w, t))


class PA:
    def __init__(self, X, y, epochs=3051):
        self.X = X
        self.y = y

        w = np.zeros((3, X[0].size))

        for _ in range(epochs):
            X, y = shuffle_xy(self.X, self.y)

            for x1, y1 in zip(X, y):
                y_hat = np.argmax(np.dot(w, x1))

                if y1 != y_hat:
                    loss = max(0, 1 - np.dot(w[y1, :], x1) + np.dot(w[y_hat, :], x1))
                    tau = loss / (2 * np.linalg.norm(x1))
                    w[y1, :] = w[y1, :] + tau * x1
                    w[y_hat, :] = w[y_hat, :] - tau * x1

        self.w = w

    def predict(self, t):
        return np.argmax(np.dot(self.w, t))


def read_files(train_x_path, train_y_path, test_x_path):
    with open(train_x_path) as f:
        X = f.read().split("\n")
        X = [np.array([float(a) for a in x.split(",")]) for x in X if len(x) > 0]
        X = np.array(X)

    with open(train_y_path) as f:
        y = f.read().split("\n")
        y = np.array([int(a) for a in y if len(a) > 0])

    with open(test_x_path) as f:
        X_test = f.read().split("\n")
        X_test = [np.array([float(a) for a in x.split(",")]) for x in X_test if len(x) > 0]
        X_test = np.array(X_test)

    return X, y, X_test


def normalize(X, X_test):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    abs_maxs = np.abs(X).max(axis=0)
    
    # Standard scaler
    # X = (X - means) / stds
    # X_test = (X_test - means) / stds
    
    # Min-Max scaler
    # X = (X - mins) / (maxs - mins)
    # X_test = (X_test - mins) / (maxs - mins)
    
    # Max-Abs scaler
    X = X / abs_maxs
    X_test = X_test / abs_maxs
    
    return X, X_test


def find_best_parameters(X, y):
    num_test = int(y.size * 0.2)
    
    train_X, train_y = X[:-num_test, :], y[:-num_test]
    test_X, test_Y = X[-num_test:, :], y[-num_test:]

    for z in [292, 355, 412, 527, 610, 741, 1150, 1468, 1618, 2168, 1201, 1908, 3051, 4293, 6041, 2508, 2540, 3004, 3204, 3340, 3284, 3204, 1236, 11392]:
        # print(f"--> Testing AVG PA - epochs = {z}")
        x = range(3)
        count = 0
        print(f"Testing {z} ")
        for n in x:
            for epochs in [z]:
                model = PA(train_X, train_y, epochs=epochs)
                x = test(model, test_X, test_Y)
                count = count + x * 100
                # print(f"epochs={epochs}, result={x}")
        count = count / 3
        if (count > 90):
            print(f"****************** PA - epochs = {z} , AVG={count}  *****************************")

    # count = 0
    # # for epochs in [438, 469, 235, 541, 542, 603, 633, 691, 746, 765, 812, 862, 872, 893, 911, 921, 927, 930, 973]:
    # for epochs in [ 893, 911, 921, 927, 930, 973]:
    #     x = range(50)
    #     for n in x:
    #         model = PA(train_X, train_y, epochs=epochs)
    #         x = test(model, test_X, test_Y)
    #         count = count + x * 100
    #         # print(f"epochs={epochs}, result={x}")
    #     count = count / 50
    #     print(f"******************AVG PA - epochs={epochs} , AVG={count}*****************************")
    #     count=0
    #     print()

    for z in range(1220, 10000, 8):
        # print(f"--> Testing AVG PA - epochs = {z}")
        x = range(20)
        count = 0
        print(f"Testing {z} ")
        for n in x:
            for epochs in [z]:
                model = PA(train_X, train_y, epochs=epochs)
                x = test(model, test_X, test_Y)
                count = count + x * 100
                # print(f"epochs={epochs}, result={x}")
        count = count / 20
        if(count > 90):
            print(f"****************** PA - epochs = {z} , AVG={count}  ***************************** ! ! ! ! ! ! ! ! ! ! !")
        z = z * 1.2



def main(train_x_path, train_y_path, test_x_path, output_log_name):
    # Open the train and test files
    X, y, X_test = read_files(train_x_path, train_y_path, test_x_path)
    
    
    # Normalize the data
    X_normalized, X_test_normalized = normalize(X, X_test)

    
    # find_best_parameters(X_normalized, y)
    

    # Remove a feature
    removed_feature = None  # None or 0 or 1 or 2 or 3 or 4
    if removed_feature is not None:
        X = np.delete(X, removed_feature, axis=1)
        X_test = np.delete(X_test, removed_feature, axis=1)


    # Run the four algorithms
    knn = KNN(X, y, 5)
    knn_results = [knn.predict(x) for x in X_test]

    perceptron = Perceptron(X_normalized, y)
    perceptron_results = [perceptron.predict(x) for x in X_test_normalized]

    svm = SVM(X_normalized, y)
    svm_results = [svm.predict(x) for x in X_test_normalized]

    pa = PA(X_normalized, y)
    pa_results = [pa.predict(x) for x in X_test_normalized]


    # Output the results
    with open(output_log_name, "w") as f:
        for i in range(len(X_test)):
            f.write(f"knn: {knn_results[i]}, perceptron: {perceptron_results[i]}, svm: {svm_results[i]}, pa: {pa_results[i]}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
