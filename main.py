
'''
- set parameters: environment type, number of players, reward matrix, num iterations, which algorithms are used
- initialise env, gather first moves from all players
- calculate move for each player based on their observations - go by average move? need to make decision on this. differentiation between who is making what move is important for watts-strogatz rewiring
- calculate reward for this step
- repeat until: convergence to stable eq, or define chaotic after set number



could create a matrix to show from who each player takes an observation from
matrix size is the number of players (-1?) ^2
fully connected is matrix of ones, different matrix created for different connection types

also have array of moves made by each player.
then,
iterate over the matrix, if there is a 1 then find the move that player made last round. use this to calculate the score
'''
from network_tops import make_fc_net, make_ring_net, make_sw_net

def initialise(env_type, num_players, player_types):
    if env_type == 'full':
        network = make_fc_net()
    if env_type == 'ring':
        network = make_ring_net()
    if env_type == 'smallworld':
        network = make_sw_net()
    return 0

def simulate_env():
    return 0