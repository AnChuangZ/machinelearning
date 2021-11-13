def loss(rx, ry, w):
    m = len(rx)
    sse = 0.
    for j in range(m):
        y = w[0] + w[1] * rx[j]
        sse += (ry[j] - y) ** 2

    #print(sse)
    #print(type(sse)) #<class 'numpy.float64'>
    return sse


def main():
    pass


if __name__ == "__main__":
    main()
