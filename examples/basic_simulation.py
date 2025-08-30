#!/usr/bin/env python3
"""
Basic NIPD Simulation Example

This example demonstrates how to use the NIPD framework to run a simple simulation
with different agent types in a small-world network.
"""

import nipd

def main():
    """Run a basic NIPD simulation"""
    
    # Configuration for a simple simulation
    agent_config = {
        'titfortat': 5,      # 5 Tit-for-Tat agents
        'cooperator': 3,     # 3 Always Cooperate agents
        'defector': 2,       # 2 Always Defect agents
        'decentralized_ppo': 0,
        'standard_mappo': 0,
        'cooperative_mappo': 0,
        'lola': 0,
        'q_learner': 0,
        'online_simple_q': 0,
        'online_q_network': 0,
        'online_decentralized_ppo': 0,
        'online_lola': 0,
        'online_mappo': 0,
    }
    
    network_config = {
        'type': 'small_world',
        'k_neighbors': 2,
        'rewire_prob': 0.1
    }
    
    simulation_config = {
        'episode_length': 100,
        'num_episodes': 1,
        'reward_matrix': [
            [3.0, 0.0],  # Cooperate vs [Cooperate, Defect]
            [5.0, 1.0]   # Defect vs [Cooperate, Defect]
        ],
        'use_system_rewards': False,
        'noise': {
            'enabled': True,
            'probability': 0.05,
            'description': 'Random chance for agents to execute opposite action than intended'
        }
    }
    
    print("Starting NIPD Basic Simulation...")
    print(f"Total agents: {sum(agent_config.values())}")
    print(f"Network type: {network_config['type']}")
    print(f"Episode length: {simulation_config['episode_length']}")
    
    # Create and run simulator
    simulator = nipd.AgentSimulator(agent_config, network_config, simulation_config)
    
    # Run simulation
    simulator.run_simulation()
    
    # Create visualizations
    simulator.create_visualizations()
    
    # Print summary
    simulator.print_summary()
    
    print("\nSimulation completed! Check the results directory for outputs.")

if __name__ == "__main__":
    main()
