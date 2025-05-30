
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
from round import simulate_round
import numpy as np
import random

def initialise_network(env_type, num_players, player_types, k_neighbours=4, p_rewire=0.1):
    network = None
    if env_type == 'full':
        network = make_fc_net(size=num_players)
    elif env_type == 'ring':
        network = make_ring_net(size=num_players, k_neighbours=k_neighbours)
    elif env_type == 'smallworld':
        network = make_sw_net(size=num_players, k_neighbours=k_neighbours, p_rewire=p_rewire)
    else:
        raise ValueError(f"Env type must be one of: 'full' 'ring' 'smallworld' ")

    player_assignments = {node_id: player_types[node_id] for node_id in range(num_players)}

    return network, player_assignments

if __name__ == "__main__":
    random.seed(99)
    NUM_PLAYERS = 100
    player_types = ['cooperator', 'defector', 'random', 'titfortat']
    players = random.choices(player_types, k=NUM_PLAYERS)
    ENV_TYPE = 'smallworld'
    MAX_ROUNDS = 100
    REWIRE_PROB = 0.3
    PD_REWARD_MATRIX = np.array([
    [3.0, 0.0],  # move = COOPERATE (0)
    [5.0, 1.0]   # move = DEFECT (1)
])

    network, player_assignments = initialise_network(ENV_TYPE, NUM_PLAYERS, players)

    print(f"Network Matrix: \n {network}")
    print(f"Players: {player_assignments}")

    all_players_last_moves = None
    cumulative_scores = np.zeros(NUM_PLAYERS, dtype=float)
    for round_num in range(1, MAX_ROUNDS):
        current_round_moves, current_round_rewards = simulate_round(
                NUM_PLAYERS,
                network,
                player_assignments,
                PD_REWARD_MATRIX,
                all_players_last_moves
            )
        all_players_last_moves = current_round_moves
        cumulative_scores += current_round_rewards
    '''
        if round_num % (MAX_ROUNDS/5) == 1:
            print(f"\n--- Round {round_num} Results ---")
            print(f"Moves: {current_round_moves} (0=Cooperate, 1=Defect)")
            print(f"Rewards: {current_round_rewards}")
    '''
    print(f"\n--- Final Results ---")
    print(f"Cumulative Scores: {cumulative_scores}")

    ranked_players = []
    for player_id in range(NUM_PLAYERS):
        player_score = cumulative_scores[player_id]
        player_label = player_assignments[player_id]
        ranked_players.append({'id': player_id, 'label': player_label, 'score': player_score})

    # Sort players by score in descending order
    ranked_players.sort(key=lambda x: x['score'], reverse=True)

    print("\n--- Ranked Order of Players ---")
    for rank, player_info in enumerate(ranked_players):
        print(f"Rank {rank + 1}: PID {player_info['id']} ({player_info['label']}) - Score: {player_info['score']:.2f}")