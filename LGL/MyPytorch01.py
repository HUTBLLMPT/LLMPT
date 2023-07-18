import torch
print(torch.version)

# ones 函数创建一个具有指定长宽的新张量，并将所有元素值设置为 1
t = torch.ones(10)
print(t)
print('t shape:', t.shape) # 访问向量的形状

# 生成一维向量，从0开始到传参前一位
x = torch.arange(12)
print('x:', x)
print('x shape:', x.shape)

y = x.reshape(3, 4)  # 改变一个张量的形状而不改变元素数量和元素值
print('y:', y)
print('y.numel():', y.numel())  # 返回张量中元素的总个数

z = torch.zeros(2, 3, 4)  # 创建一个张量，其中所有元素都设置为0
print('z:', z)

w = torch.randn(2, 3, 4)  # 每个元素都从均值为0、标准差为1的标准高斯（正态）分布中随机采样。
print('w:', w)

q = torch.tensor([[[1, 2, 3], [4, 3, 2], [7, 4, 3]], [[1, 2, 3], [4, 3, 2], [7, 4, 3]]])  # 通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
print('q:', q)
print('q shape:', q.shape)
print('q.numel():', q.numel())

# 张量的运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)  # **运算符是求幂运算
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print('cat操作 dim=0', torch.cat((X, Y), dim=0))
print('cat操作 dim=1', torch.cat((X, Y), dim=1))  # 连结（concatenate） ,将它们端到端堆叠以形成更大的张量。

print('X == Y', X == Y)  # 通过 逻辑运算符 构建二元张量
print('X < Y', X < Y)
print('张量所有元素的和:', X.sum())  # 张量所有元素的和

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]])
print('X == Y', X == Y)
print((X == Y)[0][0].type())

# 广播机制
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print('a:', a)
print('b:', b)
print('a + b:', a + b)  # 广播运算

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("X:", X)
print("X[-1]:", X[-1])
print("X[1:3]:", X[1:3])

X[1, 2] = 9  # 写入元素。
print('X:', X)

X[0:2, :] = 12  # 写入元素。
print('X:', X)


# 节约内存
before = id(Y)  # id()函数提供了内存中引用对象的确切地址
Y = Y + X
print(id(Y) == before)

before = id(X)
X += Y
print(id(X) == before)  # 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销。

before = id(X)
X[:] = X + Y
print(id(X) == before)  # 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销。


# tensor对象转换为其他 Python对象
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
A = Y.numpy()  # 转换函数
print(type(A))  # 打印A的类型
print(A)
B = torch.tensor(A)
print(type(B))  # 打印B的类型
print(B)

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))