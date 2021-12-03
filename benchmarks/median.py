import timeit
import bottleneck as bn
import numpy as np
import sys
from ndcombine import ndcombine

datamasked = None
data = None
list_of_data = None
list_of_mask = None
mask = None


def test_median(size):
    global data, mask, datamasked, list_of_data, list_of_mask

    print('Generate fake data')
    shape = (10, size, size)
    np.random.seed(42)
    data = np.random.normal(size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint16)
    data = data.reshape(shape[0], -1)
    mask = mask.reshape(shape[0], -1)
    datamasked = np.ma.array(data, mask=mask.astype(bool))
    list_of_data = list(data)
    list_of_mask = list(mask)

    print('Check results')
    outndcomb, _, _ = ndcombine(list_of_data,
                                list_of_mask,
                                combine_method='median',
                                reject_method='none')
    outnp = np.median(data, axis=0)
    outnpma = np.ma.median(datamasked, axis=0)
    outbn = bn.median(data, axis=0)

    np.testing.assert_array_equal(outndcomb, outnp)
    np.testing.assert_array_equal(outndcomb, outnpma)
    np.testing.assert_array_equal(outndcomb, outbn)

    print('Run perf tests')
    nb = 10
    kwargs = dict(globals=globals(), number=nb, repeat=5)

    def run(label, command):
        res = timeit.repeat(command, **kwargs)
        res = np.array(res) / nb
        print(f'- {label:20s}: {np.mean(res):.3f}s ± {np.std(res):.3f}s')

    run('np.median', 'np.median(data)')
    run('np.ma.median', 'np.ma.median(datamasked, axis=0)')
    run('bn.median', 'bn.median(data)')
    run(
        "ndcombine 1 thread",
        "ndcombine(list_of_data, list_of_mask, combine_method='median', "
        "reject_method='none', num_threads=1)")
    run(
        "ndcombine",
        "ndcombine(list_of_data, list_of_mask, combine_method='median', "
        "reject_method='none')")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    test_median(size)
