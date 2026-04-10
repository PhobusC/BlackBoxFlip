# This file will contain the code for the flip flopping object
import numpy as np
import random

class flipFlop():
    def __init__(self, bits=3, p=0.2):
        """
        bits: number of bits to model flipping for
        p: probabiliity of a bit flipping for one time step
        """
        random.seed(1432)
        self.bits = bits
        self.p = p

        # It would be too inefficient to create a new object for each epoch
        #init_state = [-1, 1]
        #self.state = [random.choice(init_state) for _ in range(bits)]
        

    def simNSteps(self, n_steps, batch_size):
        unsigned_in = np.random.binomial(1, self.p, [batch_size, n_steps, self.bits])
        unsigned_out = 2*np.random.binomial(1, 0.5, [batch_size, n_steps, self.bits])-1

        # Creates input impulse matrix, should be mostly 0s with occational -1s and 1s
        input = np.multiply(unsigned_in, unsigned_out)

        # Initial signal of 1 so states are set to 1
        input[0, :, 0] = 1 

        # Sustained signal


        # Initialize array of 0s for output
        # Output represents state after input signal received
        output = np.zeros_like(input)

        




        return {"inputs": input, "outputs": output}
        