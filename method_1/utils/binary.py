# see which way to get set bits faster.
# https://stackoverflow.com/questions/49592295/getting-the-position-of-1-bits-in-a-python-long-object

import timeit


def fcn1():
    for num in range(small, big):
        one_bit_indexes = []
        index = 0
        while num: # returns true if num is non-zero
            if num & 1: # returns true if right-most bit is 1
                one_bit_indexes.append(index)
            num >>= 1 # discard the right-most bit
            index += 1


# The fastest
def fcn2():
    for num in range(small, big):
        bits = []
        for i, c in enumerate(bin(num)[:1:-1], 1):
            if c == '1':
                bits.append(i)


def fcn3():
    for num in range(small, big):
        [i for i in range(num.bit_length()) if num & (1<<i)]


big = 2 ** (25*25)     # 3**10000     # 
small = big - 10000
# fcn1 is so slow. Take it out in competition.
# print(timeit.timeit(fcn1, number=1))
print(timeit.timeit(fcn2, number=1))
print(timeit.timeit(fcn3, number=1))

