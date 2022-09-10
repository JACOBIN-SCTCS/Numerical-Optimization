from scipy.optimize import rosen,rosen_der
from scipy.optimize import line_search
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.optimize import line_search
from mpl_toolkits import mplot3d





def rosenbrock(point):
    a = 1.0
    b = 100.0
    x = point[0]
    y = point[1]
    result = (1-x)**2
    result += b*((y - x**2)**2)
    return result

def rosenbrock_jacobian(point):
    a = 1.0
    b = 100.0
    x = point[0]
    y = point[1]
    result = np.zeros((2,))
    result[0] = -2.0*(a-x) - 4*b*x*(y - x**2)
    result[1] = 2.0*b*(y - x**2)
    return result

def rosenbrock_hessian(point):
    a = 1.0
    b = 100.0
    x = point[0]
    y = point[1]
    hessian = np.zeros((2,2))
    hessian[0][0] = 2*b*y - 2*b*x*x
    hessian[0][1] = -4*b*x
    hessian[1][0] = -4*b*x
    hessian[1][1] = 2*b 
    return hessian

def squared_norm(x):
    x_val = np.float64(x[0][0])
    y_val = np.float64(x[1][0])
    return x_val*x_val + y_val*y_val

def norm(x):
    return np.sqrt(squared_norm(x))

def get_unit_vector(x):
    norm_value = norm(x)
    if norm_value == np.float64(0.0):
        return x
    return (x/norm_value).copy()


def steepest_descent_rosenbrock(point):

    fun = rosenbrock
    gradient_fun = rosenbrock_jacobian
    
    current_point = point.copy()
    c = 0.0001
    rho = 0.9
    iterations = 0
    
    step_sizes = []
    iterates = []
    function_values = []
    
    iterates.append(current_point)
    function_values.append(np.float64(fun(current_point.squeeze()).squeeze()))

    while True:        
        iterations += 1
        gradient = gradient_fun(current_point)
        gradient = np.expand_dims(gradient,axis=1)
        p = -get_unit_vector(gradient)            
        alpha  = np.float64(1.0)
        previous_point = current_point.copy() 
        break_inner_loop = False
    
        current_function_value = np.float64(fun(current_point.squeeze()).squeeze()) 
        while not break_inner_loop:
            
            new_point = current_point + alpha*p

            new_function_value = np.float64(fun(new_point.squeeze()).squeeze())
            increment_value = np.float64(((c*alpha)*np.dot(gradient.T,p)).squeeze())

            if((new_function_value > (current_function_value + increment_value))):
                alpha = alpha * rho
            else:
                break_inner_loop = True
     
        step_sizes.append(alpha)
        new_point = current_point + alpha*p
        current_point = new_point.copy()
        iterates.append(current_point)
        function_values.append(np.float64(fun(new_point.squeeze()).squeeze()))

        if np.linalg.norm(current_point-previous_point) < 0.00001:
            break
    return iterations,iterates, function_values, step_sizes
    

def newton_rosenbrock(point):

    fun = rosenbrock
    gradient_fun = rosenbrock_jacobian
    hessian_fun = rosenbrock_hessian
    current_point = point.copy()
    
    iterations = 0
    function_values = []
    iterates = []
    step_sizes = []

    iterates.append(current_point)
    function_values.append(np.float64(fun(current_point.squeeze()).squeeze()))

    
    c = 0.0001
    rho = 0.9

    while True:        
        iterations+=1
        gradient = np.expand_dims(gradient_fun(current_point.squeeze()),axis=1)
        hessian = hessian_fun(current_point.squeeze())


        m = np.matmul(np.linalg.inv(hessian),gradient)
        p = -m
        print(p)
        
        break_inner_loop = False
        alpha  = np.float64(1.0)
 
        current_function_value = np.float64(fun(current_point.squeeze()).squeeze()) 
        while not break_inner_loop:

            new_point = current_point + alpha*p
            new_function_value = np.float64(fun(new_point.squeeze()).squeeze())
            increment_value = np.float64(((c*alpha)*np.dot(gradient.T,p)).squeeze())
            second_order_term = (c*(alpha*alpha/2)*(np.matmul(np.matmul(p.T,hessian),p))).squeeze()
            increment_value += np.float64(second_order_term)
            

            if((new_function_value > (current_function_value + increment_value))):
                alpha = alpha * rho
            else:
                break_inner_loop = True

        
        previous_point = current_point.copy() 
        new_point = current_point + alpha*p
        current_point = new_point.copy()
        
        step_sizes.append(alpha)
        function_values.append(np.float64(fun(new_point.squeeze()).squeeze()))
        iterates.append(current_point)
        
        if np.linalg.norm(current_point-previous_point) < 0.00001:
            break
    
    return iterations,iterates,function_values,step_sizes


fig = plt.figure()
ax = plt.axes(projection = '3d')
X = np.linspace(-1.5,1.5,500)
Y = np.linspace(-1.5,1.5,500)
X,Y = np.meshgrid(X,Y)

fun  = lambda x,y : rosenbrock(np.array([x,y]))
Z = fun(X,Y)
ax.contour3D(X,Y,Z,50)

#iterations, iterates, function_values , step_sizes = steepest_descent_rosenbrock(np.array([[1.2],[1.2]]))
print(rosenbrock(np.array([[-1.2],[0.0]])))
iterations, iterates, function_values , step_sizes= newton_rosenbrock(np.array([[-1.2],[0.0]]))
iterates = np.array(iterates)
function_values = np.array(function_values)
ax.plot3D(iterates[:,0,0],iterates[:,1,0],function_values[:])
plt.show()