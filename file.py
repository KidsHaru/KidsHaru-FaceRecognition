from functools import reduce

a = list(range(10+1))

result = reduce(lambda x, y: x+y, a)
print(result)