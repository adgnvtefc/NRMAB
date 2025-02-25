import copy
from algorithms.hillClimb import HillClimb
from algorithms.deepq import train_dqn_agent, select_action_dqn, get_dqn_total_time, get_dqn_times_called
from algorithms.whittle import WhittleIndexPolicy
from algorithms.tabularbellman import TabularBellman 
from algorithms.NEW_GraphQ import GraphQ
from algorithms.graph_env import GraphEnv
from networkSim import NetworkSim as ns
import networkx as nx
import numpy as np
import random
import timeit #for timing the runtime of various training algorithms

#Module used to compare the results of running different algorithms and saving to csv file
#ERROR: WHITTLE BREAKS IT FOR... REASONS IG
class Comparisons:

    # algorithms = {
    #     "hillclimb": run_single_hillclimb,
    #     "dqn": run_single_dqn,
    #     "whittle": run_single_whittle,
    #     "tabular": run_single_tabular,
    #     "graph": run_single_graph
    #     "none": run_single_noalg,
    #     "random": run_single_random
    # }

    ###TARGET REFACTORING
    def __init__(self):
        self.models = {}
        self.algorithms = {
            "hillclimb": self.run_single_hillclimb,
            "dqn": self.run_single_dqn,
            "whittle": self.run_single_whittle,
            "tabular": self.run_single_tabular,
            "graph": self.run_single_graph,
            "none": self.run_single_noalg,
            "random": self.run_single_random
        }
    
    def train_dqn(self, initial_graph, num_actions, cascade_prob):
        config_normal = {
                "graph": copy.deepcopy(initial_graph),
                "num_nodes": len(initial_graph.nodes),
                "cascade_prob": cascade_prob,
                #arbitrary
                "stop_percent": 0.8,
                "reward_function": "normal"
            }
        print("Training DQN agent with normal reward function...")
        #t_0 = timeit.default_timer()
        model_normal, policy_normal = train_dqn_agent(config_normal, num_actions, num_epochs=3)
        t_1 = timeit.default_timer()
        #elapsed = t_1 - t_0 #in nanoseconds
        #print(f"elapsed time: {elapsed} nanoseconds")
        self.models['dqn'] = model_normal

    def train_tabular(self, initial_graph, num_actions, gamma):
        tab_bell = TabularBellman(initial_graph, num_actions=num_actions, gamma=gamma, alpha=0.8)
            #assuming these stats don't change
        tab_bell.update_q_table(num_episodes=300, steps_per_episode=500, epsilon=0.3)

        self.models['tabular'] = tab_bell

    def train_whittle(self, initial_graph, gamma):
        transitions_whittle = {}
        node_values = {}
        for node_id in initial_graph.nodes():
            node_obj = initial_graph.nodes[node_id]['obj']
            transition_matrix = np.zeros((2, 2, 2))

            # From Passive state (s=0)
            transition_matrix[0, 0, 1] = node_obj.passive_activation_passive
            transition_matrix[0, 0, 0] = 1 - node_obj.passive_activation_passive
            transition_matrix[0, 1, 1] = node_obj.passive_activation_active
            transition_matrix[0, 1, 0] = 1 - node_obj.passive_activation_active

            # From Active state (s=1)
            transition_matrix[1, 0, 1] = node_obj.active_activation_passive
            transition_matrix[1, 0, 0] = 1 - node_obj.active_activation_passive
            transition_matrix[1, 1, 1] = node_obj.active_activation_active
            transition_matrix[1, 1, 0] = 1 - node_obj.active_activation_active

            transitions_whittle[node_id] = transition_matrix
            node_values[node_id] = node_obj.getValue()

        whittle_policy = WhittleIndexPolicy(
            transitions=transitions_whittle,
            node_values=node_values,
            discount=gamma,
            subsidy_break=0.0,
            eps=1e-4
        )

        self.models['whittle'] = whittle_policy
    
    def train_graph(self, initial_graph, num_actions, cascade_prob):
        config = {
            "graph": copy.deepcopy(initial_graph),
            "num_nodes": len(initial_graph.nodes),
            "cascade_prob": cascade_prob,
            #arbitrary
            "stop_percent": 0.8,
            "reward_function": "normal",
            "gamma": 0.8 #arbitrary
        }
        env = GraphEnv(config)
        input_dim = 7
        hidden_dim = 16
        output_dim = 1
        model = GraphQ(input_dim, hidden_dim, output_dim)
        model.train(env, num_episodes=200)
        self.models["graph"] = model



    def run_single_hillclimb(self, common_metadata, data_collection):
        graph_hill_climb = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range(1, common_metadata['timesteps'] + 1):
            seeded_nodes_hc = HillClimb.hill_climb(graph_hill_climb, num=common_metadata['num_actions'])
            ns.passive_state_transition_without_neighbors(graph_hill_climb, exempt_nodes=seeded_nodes_hc)
            ns.active_state_transition([node for node in seeded_nodes_hc])
            ns.independent_cascade_allNodes(graph_hill_climb, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_hill_climb)

            self.collect_data('hillclimb', graph_hill_climb, data_collection, timestep)
        return True
    
    #NOTE: DQN STILL BREAKS ON SMALL GRAPHS WHERE U CAN FILL OUT THE WHOLE GRAPH    
    def run_single_dqn(self, common_metadata, data_collection):
        graph_single_dqn = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range(1, common_metadata['timesteps'] + 1):
            seeded_nodes_dqn = select_action_dqn(
                graph_single_dqn,
                model=self.models['dqn'],
                num_actions=common_metadata['num_actions']
            )
            ns.passive_state_transition_without_neighbors(graph_single_dqn, exempt_nodes=seeded_nodes_dqn)
            ns.active_state_transition(seeded_nodes_dqn)
            ns.independent_cascade_allNodes(graph_single_dqn, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_single_dqn)

            self.collect_data('dqn', graph_single_dqn, data_collection, timestep)
        return True
    
    def run_single_whittle(self, common_metadata, data_collection):
        graph_single_whittle = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range(1, common_metadata['timesteps'] + 1):
            current_states_whittle = {node: int(graph_single_whittle.nodes[node]['obj'].isActive()) for node in graph_single_whittle.nodes()}
            whittle_indices = self.models['whittle'].compute_whittle_indices(current_states_whittle)
            seeded_nodes_whittle_ids = self.models['whittle'].select_top_k(whittle_indices, common_metadata['num_actions'])

            seeded_nodes_whittle = [graph_single_whittle.nodes[node_id]['obj'] for node_id in seeded_nodes_whittle_ids]
            ns.passive_state_transition_without_neighbors(graph_single_whittle, exempt_nodes=seeded_nodes_whittle)
            ns.active_state_transition(seeded_nodes_whittle)
            ns.independent_cascade_allNodes(graph_single_whittle, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_single_whittle)

            self.collect_data('whittle', graph_single_whittle, data_collection, timestep)
        return True
    
    def run_single_tabular(self, common_metadata, data_collection):
        graph_single_tabular = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range (1, common_metadata['timesteps'] + 1):
            tab_nodes, _ = self.models['tabular'].get_best_action_nodes(graph_single_tabular)
            exempt_nodes_tab = set(tab_nodes)
            ns.passive_state_transition_without_neighbors(graph_single_tabular, exempt_nodes=exempt_nodes_tab)
            ns.active_state_transition(tab_nodes)
            ns.independent_cascade_allNodes(graph_single_tabular, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_single_tabular)

            self.collect_data('tabular', graph_single_tabular, data_collection, timestep)
        return True
    
    def run_single_graph(self, common_metadata, data_collection):
        graph_single_graph = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range(1, common_metadata['timesteps'] + 1):
            seeded_nodes_graph = self.models["graph"].predict(graph_single_graph, k=common_metadata['num_actions'])

            ns.passive_state_transition_without_neighbors(graph_single_graph, exempt_nodes=seeded_nodes_graph)
            ns.active_state_transition_graph_indices(graph_single_graph, seeded_nodes_graph)
            ns.independent_cascade_allNodes(graph_single_graph, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_single_graph)

            self.collect_data('graph', graph_single_graph, data_collection, timestep)
        return True

    def run_single_noalg(self, common_metadata, data_collection):
        graph_single_noalg = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range (1, common_metadata['timesteps'] + 1):
            no_nodes = []  # empty selection
            exempt_nodes_no = set(no_nodes)
            ns.passive_state_transition_without_neighbors(graph_single_noalg, exempt_nodes=exempt_nodes_no)
            ns.active_state_transition(no_nodes)  # no nodes activated
            ns.independent_cascade_allNodes(graph_single_noalg, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_single_noalg)

            self.collect_data('none', graph_single_noalg, data_collection, timestep)
        return True

    def run_single_random(self, common_metadata, data_collection):
        graph_single_random = copy.deepcopy(common_metadata['initial_graph'])
        for timestep in range(1, common_metadata['timesteps'] + 1):
            random_nodes_indices = random.sample(range(len(graph_single_random.nodes())), common_metadata['num_actions'])
            seeded_nodes_random = [graph_single_random.nodes[node_index]['obj'] for node_index in random_nodes_indices]
            exempt_nodes = seeded_nodes_random
            ns.passive_state_transition_without_neighbors(graph_single_random, exempt_nodes=exempt_nodes)
            ns.active_state_transition(seeded_nodes_random)
            ns.independent_cascade_allNodes(graph_single_random, common_metadata['cascade_prob'])
            ns.rearm_nodes(graph_single_random)
            
            self.collect_data('random', graph_single_random, data_collection, timestep)
        return True

    #compares the performance of two or more algorithms many times
    def run_many_comparisons(self, algorithms, initial_graph, num_comparisons, num_actions, cascade_prob, gamma, timesteps, timestep_interval):
        common_metadata = {
            'initial_graph': initial_graph,
            'num_actions': num_actions,
            'cascade_prob': cascade_prob,
            'timesteps': timesteps,
            'timestep_interval': timestep_interval
        }
        trials = []            
            

        for _ in range(num_comparisons):
            result = self.run_comparisons(algorithms=algorithms, common_metadata=common_metadata)
            trials.append(result)
        if "hillclimb" in algorithms:
            print("Hill Climb")
            print(HillClimb.get_hillclimb_total_time())
            print(HillClimb.get_hillclimb_times_called())
        if "dqn" in algorithms:
            print("dqn")
            print(get_dqn_total_time())
            print(get_dqn_times_called())

        
        return trials       


    def run_comparisons(self, algorithms, common_metadata):
        data_collection = {algo: {
            'timestep': [],
            'cumulative_active_nodes': [],
            'percent_activated': [],
            'reward': [],
        } for algo in algorithms}

        for algo in algorithms:
            success = self.run_single_simulation(algo, common_metadata, data_collection)
            if not success:
                return None        
        return data_collection

    def run_single_simulation(self, algorithm, common_metadata, data_collection):
        if algorithm in self.algorithms:
            algo_func = self.algorithms[algorithm]
            return algo_func(common_metadata, data_collection)
        else:
            print("clown")
            return False
        
    
    def collect_data(self, algorithm, graph, data_collection, timestep):
        # Calculate metrics
        total_nodes = len(graph.nodes())
        active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())
        
        # Record data
        data_collection[algorithm]['timestep'].append(timestep)

        # Update cumulative active nodes (sum over time)
        if timestep == 1:
            # first time we call collect_data
            data_collection[algorithm]['cumulative_active_nodes'].append(active_nodes)
        else:
            # subsequent timesteps
            last_cumulative = data_collection[algorithm]['cumulative_active_nodes'][-1]
            data_collection[algorithm]['cumulative_active_nodes'].append(last_cumulative + active_nodes)

        # Calculate percentage of network activated (current timestep)
        percent_activated = (active_nodes / total_nodes) * 100
        data_collection[algorithm]['percent_activated'].append(percent_activated)

        data_collection[algorithm]['reward'].append(ns.reward_function(graph, seed=None))