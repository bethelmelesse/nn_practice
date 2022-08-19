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


class Var:
    def __init__(self, variable):
        self.variable = variable
        self.value = 0
        self.grad = 0

    def forward(self):
        self.value = self.variable
        return self.value

    def backward(self, incoming_grad):
        self.grad = incoming_grad + self.grad
        return incoming_grad


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

class Model:
    def __init__(self, x, y, constant):
        x = Var(x)
        y = Var(y)
        constant = Num(constant)
        a = Square(x)
        b = Multi(x, y)
        c = Square(y)
        d = Subtraction(a, b)
        e = Multi(constant, c)
        f = Addition(d, e)
        self.final_model = f
        self.x = x
        self.y = y

    def forward(self):
        return self.final_model.forward()

    def backward(self):
        return self.final_model.backward(1.0)

my_first_model = Model(2, 3, 2)
print(my_first_model.forward())
my_first_model.backward()
print(my_first_model.y.grad)
