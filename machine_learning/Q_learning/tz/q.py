from environment import MountainCar
import sys
import numpy as np
import random


# state_space; action_space; step(action)=(state,reward,done)
def main(args):
    mode = args[1]
    env = MountainCar(mode)
    env.reset()
    print(env.transform(env.state))
    print(env.reset())



if __name__ == "__main__":
    main(sys.argv)