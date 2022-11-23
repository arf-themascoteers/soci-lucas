import matplotlib.pyplot as plt
import numpy
import lucas_dataset
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
import time



def train(the_power):
    ds = lucas_dataset.LucasDataset(is_train=True, power = the_power)
    x = ds.get_x()
    y = ds.get_y()
    start = time.time()
    # print("Train started")
    reg = LinearRegression().fit(x,y)

    # print("Train done")
    end = time.time()
    required = end - start
    # print(f"Train seconds: {required}")
    return reg

    #pickle.dump(reg, open("models/linear3","wb"))

def test(reg, the_power):
    #reg = pickle.load(open('models/linear3', 'rb'))
    ds = lucas_dataset.LucasDataset(is_train=False, power = the_power)
    x = ds.get_x()
    y = ds.get_y()
    start = time.time()
    y_hat = reg.predict(x)
    end = time.time()
    required = end - start
    # print(f"Test seconds: {required}")
    r2 = r2_score(y, y_hat)
    # print("R2",r2)
    # print("MSE",mean_squared_error(y, y_hat))
    print(the_power, r2, 10000000**r2)
    return r2

def dump():
    reg = pickle.load(open('models/linear2', 'rb'))
    x = abs(reg.coef_)
    x = numpy.array(x)
    y = numpy.argsort(x)
    for i in range(len(y)):
        z = y[i]
        a = (z * 0.5) + 400
        b = z+11
        print(f"{b},",end="")
    #

def get_r2(power):
    reg = train(power)
    return test(reg, power)

x = [get_r2(i) for i in range(1,80)]
print(x)
x = numpy.array(x)
#x = x[3:]
x = 10000000**x
plt.plot(x)
plt.show()