import numpy as np

from pyhealgo import Sa

interval = (-10, 10)

def f(x):
    """ Function to minimize."""
    return x**2

def generate_initial_state(sa_instance):
    a, b = sa_instance.problem_domain
    return a + (b - a) * np.random.random_sample()

def cost_function(x):
    return f(x)

if __name__ == '__main__':

    sa_instance = Sa(
                 #generate_initial_state_hndl = generate_initial_state,
                 problem_domain=interval,
                 on_cost_hdl=cost_function,
                 max_steps=200
                 )
    state, cost = sa_instance.run()
    print("State: {0} Cost {1}".format(state,cost))
    sa_instance.visualize()
    sa_instance.save_to_file('non_quantum_random_sa')
