def square(my_input):
    return my_input ** 2


def addition(input_1, input_2):
    return input_1 + input_2

def multiplication(input_1, input_2):
    return input_1 * input_2

def subs(input_1, input_2):
    return input_1 - input_2

def prop (x, y):
    a = square(x)
    b = multiplication(x, y)
    c = square(y)
    d = subs(a, b)
    e = multiplication(2, c)
    f = addition(d, e)

    df_dd = 1
    dd_da = 1
    da_dx = 2*x
    dd_db = -1
    db_dx = y
    db_dy = x

    df_de = 1
    de_dc = 2
    dc_dy = 2*y

    df_dx_1 = df_dd * dd_da * da_dx
    df_dx_2 = df_dd * dd_db * db_dx
    df_dx = df_dx_1 + df_dx_2

    df_dy_1 = df_dd * dd_db * db_dy
    df_dy_2 = df_de * de_dc * dc_dy
    df_dy = df_dy_1 + df_dy_2

    return df_dx, df_dy


def fn(x, y):
    return x**2 - x*y + 2*y**2

def gradient_descent(x, y, step, learning_rate):
    count = 1
    x_old = x
    y_old = y
    grad_x, grad_y = prop(x_old, y_old)
    print(grad_x)
    print(grad_y)
    while count <= step:
        x_new = x_old - learning_rate * grad_x
        y_new = y_old - learning_rate * grad_y

        print(f"{count} ----- x_new = {x_new:.2f} ---- y_new = {y_new:.2f}, ----- {fn(x_new, y_new):.2f} ")
        x_old = x_new
        y_old = y_new
        count += 1


print(gradient_descent(5, 4, 100, 0.1))


