class Operators:
    def __init__(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        self.value = None


class Multi(Operators):
    def forward(self):
        self.input_1.forward()             # check here - can this be included in the superclass?
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


class Var:
    def __init__(self, variable):
        self.variable = variable
        self.value = None
        self.grad = 0

    def forward(self):
        self.value = self.variable
        return self.value

    def backward(self, incoming_grad):       # have some doubts here
        self.grad = incoming_grad + self.grad
        print("grad --- ", self.grad)
        print("variable --- ", self.variable)
        return incoming_grad


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

class Model:
    def __init__(self, x, y, constant):
        x = Var(x)
        y = Var(y)
        constant = Num(constant)

        a = Square(x, y)
        b = Multi(x, y)
        c = Square(y, x)
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


my_first_model = Model(2, 3, 2)
print(my_first_model.forward())
my_first_model.backward()
print(my_first_model.backward())
