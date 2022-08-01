
def fn(x, y):
    return x**2 - x*y + 2*y**2

def gradient_x(x, y):
    return 2*x - y

def gradient_y(x, y):
    return -x + 4*y

def gradient_descent(x, y, step, learning_rate):
    count = 1
    x_old = x
    y_old = y
    grad_x, grad_y = gradient_x(x_old, y_old), gradient_y(x_old, y_old)
    print(grad_x)
    print(grad_y)

    while count <= step:
        x_new = x_old - learning_rate * gradient_x(x_old, y_old)
        y_new = y_old - learning_rate * gradient_y(x_old, y_old)

        print(f"{count} ----- x_new = {x_new:.2f} ---- y_new = {y_new:.2f}, ----- {fn(x_new, y_new):.2f} ")
        x_old = x_new
        y_old = y_new
        count += 1


print(gradient_descent(5, 4, 100, 0.1))