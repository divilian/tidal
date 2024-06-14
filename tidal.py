
from mesa import Model, Agent
from mesa.time import RandomActivation
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import logging



class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = self.model.rng.choice(["red","blue"])
        self.extraversion = .5  # constant for all agents
        self.confidence = self.model.rng.uniform(0,1,1)[0]
    def step(self):
        logging.debug(f"Hi, I'm agent {self.unique_id}.")
        if self.model.rng.uniform(0,1,1)[0] < self.extraversion:
            neighnums = list(self.model.graph.neighbors(self.unique_id))
            neigh = self.model.schedule.agents[self.model.rng.choice(neighnums)]
            logging.debug(f" ..and I'm communicating to agent {neigh.unique_id}.")
            neigh.receive_comm(self.opinion, self.confidence)
    def receive_comm(self, opinion, confidence):
        logging.debug(f"I'm agent {self.unique_id} and I got {opinion} (with " \
            f"confidence {confidence:.2f})")


class Society(Model):
    def __init__(self, **kwd_args):
        super().__init__()
        for arg, val in kwd_args.items():
            setattr(self, arg, val)
        self.rng = np.random.default_rng(seed=138)
        self.schedule = RandomActivation(self)
        self.graph = nx.erdos_renyi_graph(self.N, .1)
        while not nx.is_connected(self.graph):
            self.graph = nx.erdos_renyi_graph(self.N, .1)
        for aid in range(self.N):
            citizen = Citizen(aid, self)
            self.schedule.add(citizen)
    def step(self):
        self.schedule.step()


parser = argparse.ArgumentParser(description="Marina model.")
parser.add_argument("-n", "--num_sims", type=int, default=1,
    help="Number of sims to run (1 = single, >1 = batch)")
parser.add_argument("-N", type=int, default=15, help="Number of agents.")
parser.add_argument("--MAX_STEPS", type=int, default=50,
    help="Maximum number of steps before simulation terminates.")

            

if __name__ == "__main__":

    # Parse arguments.
    args = parser.parse_args()

    if args.num_sims == 1:
        soc = Society(**vars(args))

    for s in range(args.MAX_STEPS):
        soc.step()
