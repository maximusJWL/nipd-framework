#!/usr/bin/env python3
"""
Advanced NIPD Simulation Example

This example demonstrates more complex configurations including:
- Multiple agent types with learning capabilities
- Different network topologies
- System vs private rewards
- Custom reward matrices
"""

import nipd

def main():
    """Run an advanced NIPD simulation"""
    
    # Advanced agent configuration with learning agents
    agent_config = {
        'titfortat': 3,
        'cooperator': 2,
        'defector': 1,
        'decentralized_ppo': 2,    # 2 PPO agents
        'standard_mappo': 2,       # 2 MAPPO agents
        'cooperative_mappo': 0,
        'lola': 0,
        'q_learner': 2,            # 2 Q-Learning agents
        'online_simple_q': 0,
        'online_q_network': 0,
        'online_decentralized_ppo': 0,
        'online_lola': 0,
        'online_mappo': 0,
    }
    
    # Ring network configuration
    network_config = {
        'type': 'ring',
        'k_neighbors': 2,
        'rewire_prob': 0.0  # Not used for ring
    }
    
    # Advanced simulation configuration
    simulation_config = {
        'episode_length': 200,
        'num_episodes': 1,
        'reward_matrix': [
            [3.0, 0.0],  # Cooperate vs [Cooperate, Defect]
            [5.0, 1.0]   # Defect vs [Cooperate, Defect]
        ],
        'use_system_rewards': True,  # Use system-wide rewards
        'noise': {
            'enabled': True,
            'probability': 0.03,
            'description': 'Low noise for realistic simulation'
        }
    }
    
    print("Starting NIPD Advanced Simulation...")
    print(f"Total agents: {sum(agent_config.values())}")
    print(f"Network type: {network_config['type']}")
    print(f"Episode length: {simulation_config['episode_length']}")
    print(f"System rewards: {simulation_config['use_system_rewards']}")
    
    # Create and run simulator
    simulator = nipd.AgentSimulator(agent_config, network_config, simulation_config)
    
    # Run simulation
    simulator.run_simulation()
    
    # Create visualizations
    simulator.create_visualizations()
    
    # Print summary
    simulator.print_summary()
    
    print("\nAdvanced simulation completed!")
    print("Check the results directory for detailed analysis.")

def compare_networks():
    """Compare different network topologies"""
    
    agent_config = {
        'titfortat': 4,
        'cooperator': 2,
        'defector': 2,
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
    
    simulation_config = {
        'episode_length': 150,
        'num_episodes': 1,
        'reward_matrix': [[3.0, 0.0], [5.0, 1.0]],
        'use_system_rewards': False,
        'noise': {'enabled': True, 'probability': 0.05}
    }
    
    network_types = ['ring', 'small_world', 'full']
    
    for network_type in network_types:
        print(f"\nRunning simulation with {network_type} network...")
        
        network_config = {
            'type': network_type,
            'k_neighbors': 2,
            'rewire_prob': 0.1 if network_type == 'small_world' else 0.0
        }
        
        simulator = nipd.AgentSimulator(agent_config, network_config, simulation_config)
        simulator.run_simulation()
        simulator.print_summary()

if __name__ == "__main__":
    print("=== Basic Advanced Simulation ===")
    main()
    
    print("\n=== Network Comparison ===")
    compare_networks()

