import numpy as np
from time import time
begin = time() 
np.random.rand(10**9).sort()
end = time()
print(f"Single processor sort time: {end - begin} seconds")