import copy
import torch
from algorithms.hillClimb import HillClimb
from algorithms.deepq import train_dqn_agent, select_action_dqn, get_dqn_total_time, get_dqn_times_called
from algorithms.whittle import WhittleIndexPolicy
from algorithms.tabularbellman import TabularBellman 
from algorithms.graphq import GraphQ
from algorithms.graphq_env import GraphEnv
from networkSim import NetworkSim as ns
import numpy as np
import random


from algorithms.cdsqn import train_cdsqn_agent, select_action_cdsqn

class Comparisons:
    def __init__(self, device=None):
        # Set device for all algorithms
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Comparisons initialized on device: {self.device}")

        self.models = {}
        self.algorithms = {
            "hillclimb": self.run_single_hillclimb,
            "dqn": self.run_single_dqn,
            "whittle": self.run_single_whittle,
            "tabular": self.run_single_tabular,
            "graph": self.run_single_graph,
            "none": self.run_single_noalg,
            "random": self.run_single_random,
            "cdsqn": self.run_single_cdsqn
        }

    def train_dqn(self, initial_graph, num_actions, cascade_prob, num_epochs=3):
        config = {
            "graph": copy.deepcopy(initial_graph),
            "num_nodes": len(initial_graph.nodes),
            "cascade_prob": cascade_prob,
            "stop_percent": 0.90,
            "reward_function": "normal"
        }
        print(f"Training DQN agent for {num_epochs} epochs...")
        model, policy = train_dqn_agent(
            config, num_actions, num_epochs=num_epochs, step_per_epoch=500
        )
        self.models['dqn'] = model.to(self.device)

    def train_cdsqn(self, initial_graph, num_actions, cascade_prob, num_epochs=3):
        config = {
            "graph": copy.deepcopy(initial_graph),
            "num_nodes": len(initial_graph.nodes),
            "cascade_prob": cascade_prob,
            "stop_percent": 0.90,
            "reward_function": "normal"
            # num_actions and gamma removed as they are handled by agent params or hardcoded env
        }
        print(f"Training CDSQN agent for {num_epochs} epochs...")
        model, policy = train_cdsqn_agent(
            config, num_actions, num_epochs=num_epochs, step_per_epoch=500 # Use same epochs as DQN for fair comparison
        )
        self.models['cdsqn'] = model.to(self.device)

    def train_tabular(self, initial_graph, num_actions, gamma):
        tab = TabularBellman(
            initial_graph, num_actions=num_actions,
            gamma=gamma, alpha=0.8
        )
        tab.update_q_table(num_episodes=300, steps_per_episode=500, epsilon=0.1)
        self.models['tabular'] = tab

    def train_whittle(self, initial_graph, gamma):
        transitions, values = {}, {}
        for nid, data in initial_graph.nodes(data=True):
            obj = data['obj']
            tm = np.zeros((2,2,2))
            tm[0,0] = [1-obj.passive_activation_passive, obj.passive_activation_passive]
            tm[0,1] = [1-obj.passive_activation_active, obj.passive_activation_active]
            tm[1,0] = [1-obj.active_activation_passive, obj.active_activation_passive]
            tm[1,1] = [1-obj.active_activation_active, obj.active_activation_active]
            transitions[nid] = tm
            values[nid] = obj.getValue()

        policy = WhittleIndexPolicy(
            transitions=transitions,
            node_values=values,
            discount=gamma,
            subsidy_break=0.0,
            eps=1e-2,
            device=self.device
        )
        self.models['whittle'] = policy

    def train_graph(self, initial_graph, num_actions, cascade_prob, gamma):
        config = {
            "graph": copy.deepcopy(initial_graph),
            "num_nodes": len(initial_graph.nodes),
            "cascade_prob": cascade_prob,
            "stop_percent": 0.90,
            "reward_function": "normal",
            "gamma": gamma
        }
        env = GraphEnv(config)
        model = GraphQ(input_dim=10, hidden_dim=32, output_dim=1, gamma=config['gamma'])
        model.to(self.device)
        model.train(env, num_episodes=100, save_path='results/rewards.png')
        self.models['graph'] = model

    def run_single_hillclimb(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            seeds = HillClimb.hill_climb(g, num=md['num_actions'])
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=seeds)
            ns.active_state_transition(seeds)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('hillclimb', g, data, t)
        return True

    def run_single_dqn(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            seeds = select_action_dqn(
                g, model=self.models['dqn'], num_actions=md['num_actions']
            )
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=seeds)
            ns.active_state_transition(seeds)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('dqn', g, data, t)
        return True
    
    def run_single_cdsqn(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        
        for t in range(1, md['timesteps']+1):
            # Select action using agent's greedy hill-climber (inference function)
            seeds = select_action_cdsqn(
                g, model=self.models['cdsqn'], num_actions=md['num_actions']
            )
            
            # Seeds are already node objects returned by select_action_cdsqn
            
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=seeds)
            ns.active_state_transition(seeds)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('cdsqn', g, data, t)
        return True

    def run_single_whittle(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            states = {n: int(g.nodes[n]['obj'].isActive()) for n in g}
            idx = self.models['whittle'].compute_whittle_indices(states)
            picks = self.models['whittle'].select_top_k(idx, md['num_actions'])
            objs = [g.nodes[i]['obj'] for i in picks]
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=objs)
            ns.active_state_transition(objs)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('whittle', g, data, t)
        return True

    def run_single_tabular(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            seeds,_ = self.models['tabular'].get_best_action_nodes(g)
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=seeds)
            ns.active_state_transition(seeds)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('tabular', g, data, t)
        return True

    def run_single_graph(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            seeds = self.models['graph'].predict(g, k=md['num_actions'])
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=seeds)
            ns.active_state_transition_graph_indices(g, seeds)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('graph', g, data, t)
        return True

    def run_single_noalg(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=[])
            ns.active_state_transition([])
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('none', g, data, t)
        return True

    def run_single_random(self, md, data):
        g = copy.deepcopy(md['initial_graph'])
        for t in range(1, md['timesteps']+1):
            idx = random.sample(range(len(g.nodes())), md['num_actions'])
            objs = [g.nodes[i]['obj'] for i in idx]
            ns.passive_state_transition_without_neighbors(g, exempt_nodes=objs)
            ns.active_state_transition(objs)
            ns.independent_cascade_allNodes(g, md['cascade_prob'])
            ns.rearm_nodes(g)
            self.collect_data('random', g, data, t)
        return True

    def run_many_comparisons(self, algorithms, initial_graph, num_comparisons, num_actions, cascade_prob, gamma, timesteps, timestep_interval, device=None):
        common = {
            'initial_graph': initial_graph,
            'num_actions': num_actions,
            'cascade_prob': cascade_prob,
            'timesteps': timesteps,
            'timestep_interval': timestep_interval
        }
        trials=[]
        for _ in range(num_comparisons):
            trials.append(self.run_comparisons(algorithms, common))
        if 'dqn' in algorithms:
            print("DQN total time:", get_dqn_total_time())
        return trials

    def run_comparisons(self, algos, common):
        data={a:{'timestep':[], 'cumulative_active_nodes':[], 'percent_activated':[], 'reward':[]} for a in algos}
        for a in algos:
            if not self.algorithms[a](common, data):
                return None
        return data

    def collect_data(self, alg, graph, data, t):
        total=len(graph.nodes())
        active=sum(1 for n in graph if graph.nodes[n]['obj'].isActive())
        data[alg]['timestep'].append(t)
        cum_prev=data[alg]['cumulative_active_nodes'][-1] if data[alg]['cumulative_active_nodes'] else 0
        data[alg]['cumulative_active_nodes'].append(cum_prev+active)
        data[alg]['percent_activated'].append((active/total)*100)
        data[alg]['reward'].append(ns.reward_function(graph, seed=None))
