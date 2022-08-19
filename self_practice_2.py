class Operators:
    def __init__(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        self.value = None


class Multi(Operators):
    def forward(self):
        self.input_1.forward()  # check here - can this be included in the superclass?
        self.input_2.forward()
        self.value = self.input_1.value * self.input_2.value
        return self.value

    def backward(self, incoming_grad):
        grad_1 = self.input_2.value
        grad_2 = self.input_1.value
        grad_1 = grad_1 * incoming_grad
        grad_2 = grad_2 * incoming_grad
        self.input_1.backward(grad_1)
        self.input_2.backward(grad_2)
        return grad_1, grad_2

    def reset_value(self):
        self.value = None
        self.input_1.reset_value()
        self.input_2.reset_value()


class Addition(Operators):
    def forward(self):
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


class Subtraction(Operators):
    def forward(self):
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


class Square(Operators):
    def forward(self):
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
        self.value = self.variable
        return self.value

    def backward(self, incoming_grad):
        self.grad = incoming_grad + self.grad
        # print("grad --- ", self.grad,"        variable --- ", self.variable)
        return incoming_grad

    def reset_value(self):
        self.value = None


class Num:
    def __init__(self, constant):
        self.constant = constant
        self.value = None

    def forward(self):
        self.value = self.constant
        return self.value

    def backward(self, incoming_grad):
        grad = 0
        return grad

    def reset_value(self):
        return


class Model:
    def __init__(self, x, y, constant):
        x = Var(x)
        y = Var(y)
        constant = Num(constant)

        a = Square(x, y)
        b = Multi(x, y)
        c = Square(y, x)  # check here - one variable
        d = Subtraction(a, b)
        e = Multi(constant, c)
        f = Addition(d, e)
        self.f = f
        self.x = x
        self.y = y

    def forward(self):
        return self.f.forward()

    def backward(self):
        return self.f.backward(1)

    def zero_grad(self):
        self.x.grad = 0
        self.y.grad = 0
        self.f.reset_value()


# my_first_model = Model(2, 3, 2)
#
# print(my_first_model.forward())
# print(my_first_model.backward())
# print(my_first_model.x.grad)
# print(my_first_model.y.grad)
#
# my_first_model.x.variable = 4
# my_first_model.y.variable = 5
#
# # my_first_model.x.grad = 0
# # my_first_model.y.grad = 0
# my_first_model.zero_grad()
# print(my_first_model.forward())
# print(my_first_model.backward())
# print(my_first_model.x.grad)
# print(my_first_model.y.grad)

def gradient_descent(x, y, constant, step, learning_rate):
    my_model = Model(x, y, constant)
    count = 1

    while count <= step:
        my_model.zero_grad()
        my_model.forward()
        my_model.backward()

        x_old = my_model.x.variable
        y_old = my_model.y.variable

        grad_x = my_model.x.grad
        grad_y = my_model.y.grad

        x_new = x_old - learning_rate * grad_x
        y_new = y_old - learning_rate * grad_y

        my_model.x.variable = x_new
        my_model.y.variable = y_new

        print(f"{count} ----- x_new = {x_new:.2f} ---- y_new = {y_new:.2f}, ----- {my_model.forward():.2f} ")
        count += 1


gradient_descent(5, 4, 2, 100, 0.1)
