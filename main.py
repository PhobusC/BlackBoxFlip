import numpy as np

batch_size = 64
time = 500
bits = 3
p = 0.2

unsigned_inp = np.random.binomial(1,p,[batch_size,time//10,bits])
unsigned_out = 2*np.random.binomial(1,0.5,[batch_size,time//10,bits]) -1 

inputs = np.multiply(unsigned_inp,unsigned_out)
print(inputs)
print()
inputs[:,0,:] = 1
print(inputs)
print()
inputs = np.repeat(inputs,10,axis=1)
output = np.zeros_like(inputs)

print(inputs[0][:30])
print()
