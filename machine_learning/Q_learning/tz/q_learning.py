from environment import MountainCar
import sys
import numpy as np
import random
import matplotlib.pyplot as plt


# state_space; action_space; step(action)=(state,reward,done)
def main(args):
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])

    beta = 0.96
    tmp = 0.0
    vt = np.array([], dtype="float64")
    returns_list = np.array([], dtype="float64")

    env = MountainCar(mode)
    S_size = env.state_space
    A_size = env.action_space
    W = np.zeros([S_size, A_size], dtype="float64")
    # print(W.shape)
    b = 0
    parameters = {"W": W, "b": b}

    with open(returns_out, "w") as fout:
        for i in range(episodes):
            env.reset()
            state = env.transform(env.state)
            # print(state)
            returns = 0.0
            done = False
            for j in range(max_iterations):
                Q = Q_calculation(state, parameters)
                # print(Q)
                a = find_action(epsilon, Q, A_size)
                grads, reward, state, done = grads_calculation(parameters, state, a, env, Q, gamma)
                parameters = update(grads, parameters, learning_rate)
                returns += reward

                if done != False:
                    break
            returns_list = np.append(returns_list, returns)
            fout.write(str(returns) + "\n")
            tmp = (beta * tmp + (1 - beta) * returns)
            tmp1 = tmp / (1 - beta ** (i + 1))

            vt = np.append(vt, tmp1)
    # print(vt)

    x = range(1, episodes + 1)
    m = plt.plot(x, returns_list)
    n = plt.plot(x, vt)
    plt.legend(('Returns', 'Rolling Mean'), loc='upper left')
    plt.title("tile mode: returns and rolling mean")
    plt.ylabel("returns & rolling mean")
    plt.xlabel("epochs")
    plt.show()

    write_weights(parameters, weight_out)


def find_action(epsilon, Q, A_size):
    act = 0
    x = random.random()
    act = np.where(x > epsilon, np.argmax(Q), random.randint(0, A_size - 1))
    return act


def Q_calculation(state, parameters):
    Q = np.array([], dtype="float64")
    W = parameters["W"]
    b = parameters["b"]
    # print(W.shape[1])

    for j in range(W.shape[1]):
        res = 0.0
        for key, val in state.items():
            # print(W[int(key)][j])
            res += val * W[int(key)][j]
        res += b
        Q = np.append(Q, float(res))
    return Q


def grads_calculation(parameters, state, a, env, Q, gamma):
    grads = {}
    new_s, r, done = env.step(a)
    new_Q = Q_calculation(new_s, parameters)
    TD_target = r + gamma * np.max(new_Q)
    TD_Error = Q[a] - TD_target

    dW = np.zeros_like(parameters["W"], dtype="float64")
    for k, v in state.items():
        dW[k][a] = v * TD_Error
    db = TD_Error

    grads["dW"] = dW
    grads["db"] = db

    return grads, r, new_s, done


def update(grads, parameters, learning_rate):

    parameters["W"] -= learning_rate * grads["dW"]
    parameters["b"] -= learning_rate * grads["db"]

    return parameters


def write_weights(parameters, weight_out):
    W = parameters["W"]
    b = parameters["b"]
    with open(weight_out, "w") as outf:
        W = W.reshape(-1,)
        W = np.append(b, W)
        for item in W:
            outf.write(str(item) + "\n")


if __name__ == "__main__":
    main(sys.argv)