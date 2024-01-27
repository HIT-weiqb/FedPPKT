import numpy as np
import random
if __name__ == '__main__':
    ave = [0.504, 0.984, 1.512, 1.928, 2.112, 2.212, 2.104, 2.208]
    A = [0.5688,0.9362,1.5316,2.0152,2.1278,2.1841,2.1744,2.2238]
    B = []
    for i in range(len(ave)):
        tmp = ave[i]*2 - A[i]
        tmp = round(tmp,3)
        B.append(tmp)
        print(B[i])
    # for i in range(len(ave)):
    #     arr[i] = arr[i] + random.uniform(-0.1,0.1)
    #     arr[i] = round(arr[i],4)
    #     print(arr[i])