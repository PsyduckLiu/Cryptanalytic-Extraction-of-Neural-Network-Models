
def binary_search(offset=None, direction=None, DIM=None, low=-1e3, high=1e3):
    """
    A slow but correct implementation of binary search to identify critical points.
    Just performs binary searcon from [low,high] splitting down the middle any time
    we determine it's not a linear function.

    In practice we perform binary search between the points
    (offset + direction * low)  ->  (offset + direction * high)
    """

    if offset is None:
        offset = np.random.normal(0,1,size=(DIM))
    if direction is None:
        direction = np.random.normal(0,1,size=(DIM))

    c = {}
    def memo_forward_pass(x):
        if x not in c:
            c[x] = run((offset+direction*x)[np.newaxis,:])
        return c[x]
    
    relus = []

    def search(low, high):
        mid = (low+high)/2

        y1 = f_low = memo_forward_pass(low) 
        f_mid = memo_forward_pass(mid) 
        y2 = f_high = memo_forward_pass(high)

        if np.abs(f_mid - (f_high + f_low)/2)/(high-low) < 1e-8:
            # In a linear region
            return
        elif high-low < 1e-6:
            # In a non-linear region and the interval is small enough
            relus.append(offset + direction*mid)
            return

        search(low, mid)
        if len(relus) > 0:
            # we're done because we just want the left-most solution; don't do more searching
            return
        search(mid, high)

    search(np.float64(low), np.float64(high))

    return relus