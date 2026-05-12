import numpy as np
import matplotlib.pyplot as plt
import sklearn
from force import ForceLearner, Reservoir
from flipflop import FlipFlop2
import torch
import time


def train(verbose=False):
    N = 1000
    d_in = 3
    d_out = 3
    n_steps = 10000
    batch_size = 1

    states = []

    model = Reservoir(d_in=d_in, d_out=d_out, N=N)
    learner = ForceLearner(model)
    flip = FlipFlop2()

    inp, out = flip.genData(n_steps=n_steps, batch_size=batch_size)

    t0 = time.time()
    
    for b in range(batch_size):
        if verbose:
            print(f"Batch {b+1}:")

        batch_in = inp[b]
        batch_out = out[b]

        for t in range(n_steps):
            if verbose:
                print(f"Iteration {t+1} / {n_steps}")

            cur_in = torch.tensor(batch_in[t]).unsqueeze(1)
            cur_out = torch.tensor(batch_out[t]).unsqueeze(1)

            z_pre = model.forward()

            z_post = learner.step(z_pre, cur_out)

            model.update(cur_in, z_post)

            states.append(model.x)

        if verbose:
            print(f"Batch {b+1} training complete")

    print("Training success!")
    print(f"Time for training {n_steps} steps: {time.time()-t0} seconds")

    return model, learner, states

def main():
    model, learner, states = train()

    def q(x):
        """
        F = -1 * x + J_layer @ r + Jf_layer @ z_post + B @ inp, but z_post set to 0
        q = 1/2|F|^2
        """
        f = -x + (model.J_layer + model.Jf_layer @ model.w) @ model.r
        return 0.5 * f.T @ f
    

    
        




if __name__ == "__main__":
    main()