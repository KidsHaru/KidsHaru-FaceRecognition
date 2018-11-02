import sys
import numpy as np
from utility import path

str1 = path.getDirname("image")
print(str1)


'''
num = np.array([1, 2, 3, 4, 5], dtype='int64')

num = np.append(num, 6)

print(num)
'''

'''
if __name__ == "__main__":
    if len(sys.argv) - 1:
        if(sys.argv[1] == '-d' or sys.argv[1] == '--download'):
            print('다운로드 시작')
        else:
            print('명령 인자를 바르게 입력해주세요')

    else:
        print('명령 인자를 바르게 입력해주세요')
'''