# Gradient Descent using Newton's method
import numpy as np
from scipy.optimize import line_search

a=1
b = 100
def Jacobian(x):
    #return array([.4*x[0],2*x[1]])
    return np.array([-2*(a-x[0]) - 2*b*(x[1]-x[0]*x[0])*2*x[0] ,  2*b*(x[1]-x[0]*x[0])])

def Hessian(x):
    #return array([[.2,0],[0,1]])
    #return np.array([[10,8],[8,10]])
    return np.array([[2*b*x[1]-2*b*x[0]*x[0], -4*b*x[0]],[-4*b*x[0], 2*b]])

def f(x):
    return (a-x[0])**2 + b*(x[1] - x[0]*x[0])**2 

def Newton(x0):

    i = 0
    iMax = 30
    x = x0
    Delta = 1
    alpha = 1
    
    while i<iMax and Delta>10**(-10):
        p = -np.dot(np.linalg.inv(Hessian(x)),Jacobian(x))
        print(p)
        xOld = x
       
        step_length = line_search(f,Jacobian,x,p)
        if(step_length[0] == None):
            break
        x = x + step_length[0]*p
    
        Delta = np.sum((x-xOld)**2)
        print ("Iteration"+str(i+1))
        print ("    x="+str(x)+"  "+"Delta="+str(Delta))
        if Delta <=10**(-10):
            print ("Performance goal achieved")
        i += 1
        
        if i == iMax:
            print ("Maximum iterations achieved")
    print (x)
    
x0 = np.array([-1.2,1.0])
Newton(x0)