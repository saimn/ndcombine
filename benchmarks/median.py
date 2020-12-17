import timeit
import bottleneck as bn
import numpy as np
from ndcombine import ndcombine

data = None
mask = None
datamasked = None


def test_median():
    global data, mask, datamasked

    shape = (10, 1000, 1000)
    np.random.seed(42)
    data = np.random.normal(size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint16)
    data = data.reshape(shape[0], -1)
    mask = mask.reshape(shape[0], -1)
    datamasked = np.ma.array(data, mask=mask.astype(bool))

    outndcomb, _, _ = ndcombine(data,
                                mask,
                                combine_method='median',
                                reject_method='none')
    outnp = np.median(data, axis=0)
    outnpma = np.ma.median(datamasked, axis=0)
    outbn = bn.median(data, axis=0)

    np.testing.assert_array_equal(outndcomb, outnp)
    np.testing.assert_array_equal(outndcomb, outnpma)
    np.testing.assert_array_equal(outndcomb, outbn)

    nb = 10
    kwargs = dict(globals=globals(), number=nb, repeat=5)

    def run(label, command):
        res = timeit.repeat(command, **kwargs)
        res = np.array(res) / nb
        print(f'{label:20s}: {np.mean(res):.3f}s ± {np.std(res):.3f}s')

    run('np.median', 'np.median(data)')
    run('np.ma.median', 'np.ma.median(datamasked, axis=0)')
    run('bn.median', 'bn.median(data)')
    run(
        'ndcombine 1 thread', "ndcombine(data, mask, combine_method='median', "
        "reject_method='none', num_threads=1)")
    run(
        'ndcombine', "ndcombine(data, mask, combine_method='median', "
        "reject_method='none')")


if __name__ == "__main__":
    test_median()
