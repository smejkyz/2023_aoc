import functools


@functools.lru_cache(maxsize=None)
def fib(x):
    print(x)
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return fib(x-1) + fib(x-2)


if __name__ == '__main__':
    out = fib(5)
    print("---")
    print(out)