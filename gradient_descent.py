
def fn(x):
    return x**2 + x + 1

def gradient(x):
    return 2*x + 1


def gradient_descent(x, step, learning_rate):
    count = 1
    x_old = x
    while count <= step:
        x_new = x_old - learning_rate * gradient(x_old)
        print(count, "------", x_new, "--------", fn(x_new))
        x_old = x_new
        count += 1


print(gradient_descent(0, 100, 0.1))