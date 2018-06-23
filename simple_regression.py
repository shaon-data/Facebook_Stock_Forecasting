from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

x = np.array([1,2,3,4,5,6],dtype=np.float64)
y = np.array([5,4,6,5,6,7],dtype=np.float64)

def create_dataset(n, variance, step=2, correlation='pos'):
    val = 1
    y = []
    for i in range(n):
        y.append(  val + random.randrange(-variance, variance)  )
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    x = [i for i in range(len(y))]
    return np.array(x , dtype=np.float64), np.array(y , dtype=np.float64)
    
def m_b(x,y):
    m = ((mean(x)*mean(y)) - mean(x*y)) / ((mean(x)**2) - mean(x**2))
    b = mean(y)-m*mean(x)
    return m,b

def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(y,y_):
    squared_error_y_reg = squared_error(y, y_)
    squared_error_y_mean = squared_error(y, mean(y) )
    return 1 - (squared_error_y_reg/squared_error_y_mean)

#x,y = create_dataset(40, 80, 2, correlation=False)
    
m,b= m_b(x,y)
y_ = m*x + b

# prediction
prediction_x = 6.5
prediction_y = m*prediction_x + b

r_squared = coefficient_of_determination(y,y_)
print("Accuracy %s"%r_squared)

plt.scatter(x,y, label='real data points', color='r')
plt.scatter(prediction_x,prediction_y, label='single prediction', color='g')
plt.plot(x,y_, label='prediction line(y_hat)')

plt.legend(loc=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression(SLR)')
plt.tight_layout()
plt.savefig('pics/simple_linear_regression1.png')
plt.show()


# error line plotting
plt.scatter(x,y, label='real data points', color='r')
plt.plot(x,y_, label='prediction line')
i=0

for xi,y_i,yi in zip(np.nditer(x),np.nditer(y_),np.nditer(y)):    
    plt.plot([xi,xi],[y_i,yi],label='error',color='b')
    if i == 0:
        i=1
        plt.legend(['Prediction line (y_hat)', 'Prediction Error(y_hat-y)' , 'Data Points'],loc=4)
        
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Errors of Prediction SLR')
plt.tight_layout()
plt.savefig('pics/errors1.png')
plt.show()
