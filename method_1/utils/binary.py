# see which way to get set bits faster.
# https://stackoverflow.com/questions/49592295/getting-the-position-of-1-bits-in-a-python-long-object

import timeit


def fcn1():
    sum = 3**100000
    one_bit_indexes = []
    index = 0
    while sum: # returns true if sum is non-zero
        if sum & 1: # returns true if right-most bit is 1
            one_bit_indexes.append(index)
        sum >>= 1 # discard the right-most bit
        index += 1
    return one_bit_indexes


def fcn2():
    number = 3**100000
    bits = []
    for i, c in enumerate(bin(number)[:1:-1], 1):
        if c == '1':
            bits.append(i)
    return bits


def fcn3():
    sum = 3**100000
    return [i for i in range(sum.bit_length()) if sum & (1<<i)]


print(timeit.timeit(fcn1, number=1))
print(timeit.timeit(fcn2, number=1))
print(timeit.timeit(fcn3, number=1))

