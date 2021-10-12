import time


def speed_test(slow_func, fast_func, *func_inputs):
    start_ = time.time()
    res1 = slow_func(*func_inputs)
    end_ = time.time()
    print("Elapsed (without numba jit) = %s" % (end_ - start_))

    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.time()
    res2 = fast_func(*func_inputs)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))
    print(
        "Percentage speedup = %s"
        % ((((end_ - start_) - (end - start)) / (end_ - start_)) * 100)
    )

    # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    start = time.time()
    res3 = fast_func(*func_inputs)
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))
    print(
        "Percentage speedup = %s"
        % ((((end_ - start_) - (end - start)) / (end_ - start_)) * 100)
    )

    return res1, res2, res3
