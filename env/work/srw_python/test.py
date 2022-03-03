import cupy as cp
import time

def pool_stats(mempool):
    print('used:',mempool.used_bytes(),'bytes')
    print('total:',mempool.total_bytes(),'bytes\n')

pool = cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed) # get unified pool
cp.cuda.set_allocator(pool.malloc) # set unified pool as default allocator

print('create first variable')
val1 = cp.zeros((50*1024,10*1024))
pool_stats(pool)

print('create second variable')
val2 = cp.zeros((50*1024,10*1024))
pool_stats(pool)

print('delete first variable')
del val1
pool_stats(pool)

print('delete second variable')
del val2
pool_stats(pool)

print('free cupy memory')
pool.free_all_blocks()
pool_stats(pool)

print('free cupy memory')
pool.free_all_blocks()
pool_stats(pool)
