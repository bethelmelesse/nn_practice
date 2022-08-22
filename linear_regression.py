class Multi:
    def __init__(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        self.value = None

    def forward(self):
        if self.value is None:
            self.input_1.forward()
            self.input_2.forward()
            self.value = self.input_1.value * self.input_2.value
        return self.value

    def backward(self, incoming_grad):
        grad_1, grad_2 = self.input_2.value, self.input_1.value
        grad_1 = grad_1 * incoming_grad
        grad_2 = grad_2 * incoming_grad
        self.input_1.backward(grad_1)
        self.input_2.backward(grad_2)
        return grad_1, grad_2

    def reset_value(self):
        self.value = None
        self.input_1.reset_value()
        self.input_2.reset_value()


class Addition:
    def __init__(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        self.value = None

    def forward(self):
        if self.value is None:
            self.input_1.forward()
            self.input_2.forward()
            self.value = self.input_1.value + self.input_2.value
        return self.value

    def backward(self, incoming_grad):
        grad = 1.0
        grad = grad * incoming_grad
        self.input_1.backward(grad)
        self.input_2.backward(grad)
        return grad

    def reset_value(self):
        self.value = None
        self.input_1.reset_value()
        self.input_2.reset_value()


class Subtraction:
    def __init__(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        self.value = None

    def forward(self):
        if self.value is None:
            self.input_1.forward()
            self.input_2.forward()
            self.value = self.input_1.value - self.input_2.value
        return self.value

    def backward(self, incoming_grad):
        grad_1 = 1.0
        grad_2 = -1.0
        grad_1 = grad_1 * incoming_grad
        grad_2 = grad_2 * incoming_grad
        self.input_1.backward(grad_1)
        self.input_2.backward(grad_2)
        return grad_1, grad_2

    def reset_value(self):
        self.value = None
        self.input_1.reset_value()
        self.input_2.reset_value()


class Square:
    def __init__(self, input_1):
        self.input_1 = input_1
        self.value = None

    def forward(self):
        if self.value is None:
            self.input_1.forward()
            self.value = self.input_1.value ** 2
        return self.value

    def backward(self, incoming_grad):
        grad = 2 * self.input_1.value
        grad = grad * incoming_grad
        self.input_1.backward(grad)
        return grad

    def reset_value(self):
        self.value = None
        self.input_1.reset_value()


class Var:
    def __init__(self, variable):
        self.variable = variable
        self.value = None
        self.grad = 0

    def forward(self):
        if self.value is None:
            self.value = self.variable
        return self.value

    def backward(self, incoming_grad):
        self.grad = incoming_grad + self.grad
        return incoming_grad

    def reset_value(self):
        self.value = None


class Num:
    def __init__(self, constant):
        self.constant = constant
        self.value = 0

    def forward(self):
        self.value = self.constant
        return self.value

    def backward(self, incoming_grad):
        grad = 0
        return grad

    def reset_value(self):
        return


class Model:
    def __init__(self, x, y, c_1, c_2, c_3, c_4):
        x = Var(x)
        y = Var(y)
        c_1 = Num(c_1)
        c_2 = Num(c_2)
        c_3 = Num(c_3)
        c_4 = Num(c_4)

        a = Square(x)
        b = Multi(a, c_1)
        c = Multi(x, c_2)
        d = Subtraction(b, c)
        e = Multi(c_3, x)
        f = Multi(e, y)
        g = Addition(f, d)
        h = Square(y)
        i = Addition(g, h)
        j = Subtraction(i, y)
        k = Addition(j, c_4)
        self.final_model = k
        self.x = x
        self.y = y

    def forward(self):
        return self.final_model.forward()

    def backward(self):
        return self.final_model.backward(1.0)

    def zero_grad(self):
        self.x.grad = 0
        self.y.grad = 0
        self.final_model.reset_value()


def gradient_descent(x, y, c_1, c_2, c_3, c_4, step, learning_rate):
    count = 1
    my_first_model = Model(x, y, c_1, c_2, c_3, c_4)

    while count <= step:
        my_first_model.zero_grad()
        my_first_model.forward()
        my_first_model.backward()
        grad_x = my_first_model.x.grad
        grad_y = my_first_model.y.grad
        # print("grad_x = ", grad_x)
        # print("grad_y = ", grad_y)

        x_new = my_first_model.x.variable - learning_rate * grad_x
        y_new = my_first_model.y.variable - learning_rate * grad_y

        my_first_model.x.variable = x_new
        my_first_model.y.variable = y_new

        print(f"{count} ----- x_new = {x_new:.2f} ---- y_new = {y_new:.2f}, ----- {my_first_model.forward():.2f} ")
        # print(f"{count} ----- x_new = {x_new:.2f} ---- y_new = {y_new:.2f},")
        count += 1


print(gradient_descent(0, 0, 7.5, 3.5, 5, 0.5, 300, 0.1))