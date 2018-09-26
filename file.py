# ================================================================
# all, any 함수 구현
# built-in function인 all, any함수를 구현해보세요.

print(all([True, False, False]))
print(all([True, True, True]))
print(any([True, False, False]))
print(any([True, True, True]))

# ================================================================
# 재귀 함수
# 다음을 참고하여 재곱근의 근사값을 구하는 재귀함수를 만들어보세요.

def mySelf(x, y):
    if abs(y - x**2) <= 1e-7:
        return x
    else:
        x = (x + (y/x)) / 2
        return mySelf(x, y)

print(mySelf(2, 28))

# ================================================================
# 재귀 함수
# 다음을 참고하여 재곱근의 근사값을 구하는 재귀함수를 만들되,
# “한 없이 가깝다면” 을 “전에 값과 변함이 없다면" 으로 조건을 바꿔 만들어보세요

def mySelf(x, y):
    num = x
    x = (x + (y/x)) / 2

    if x == num:
        return x
    else:
        return mySelf(x, y)

print(mySelf(2, 28))


# ================================================================
# isnumeric() 함수 구현
# string에는 숫자로 변환 가능한 string인지 알려주는 isnumeric()메소드가 있습니다,
# 예 ) >>> ‘1.234’.isnumeric()
# => True
# >>> ‘1.2ab’.isnumeric()
# => False
# isnumeric()함수를 구현해보세요
# 예 ) >>> isnumeric(‘1.234’)
# => True

print('1234'.isnumeric())
print('1.234'.isnumeric())
print('1.2ab'.isnumeric())

def isnumeric(num):
    try:
        if float(num):
            return True
        
    except ValueError:
        return False

print(isnumeric("1234"))