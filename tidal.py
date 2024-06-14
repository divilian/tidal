
from mesa import Model, Agent
from mesa.time import BaseScheduler
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
        self.base_confidence = self.model.rng.uniform(0,1,1)[0]
        self.confidence = self.base_confidence
    def step(self):
        logging.info(f"Hi, I'm agent {self.unique_id}.")
        if self.model.rng.uniform(0,1,1)[0] < self.extraversion:
            neighnums = list(self.model.graph.neighbors(self.unique_id))
            neigh = self.model.schedule.agents[self.model.rng.choice(neighnums)]
            logging.info(f" ..and I'm talking at agent {neigh.unique_id}.")
            neigh.receive_comm(self.opinion, self.confidence)
    def receive_comm(self, neigh_opinion, neigh_confidence):
        logging.info(f"I'm agent {self.unique_id} and I got {neigh_opinion} "\
            f"(with confidence {neigh_confidence:.2f})")
        if self.believe(neigh_opinion, neigh_confidence):
            if self.opinion != neigh_opinion:
                self.opinion = neigh_opinion
                self.confidence = self.base_confidence
                self.model.display()
    def believe(self, neigh_opinion, neigh_confidence):
        return True


class Society(Model):
    def __init__(self, **kwd_args):
        super().__init__()
        for arg, val in kwd_args.items():
            setattr(self, arg, val)
        self.rng = np.random.RandomState(seed=self.seed)
        self.schedule = BaseScheduler(self)
        self.fig, self.ax = plt.subplots()
        self.iter = 1
        self.graph = nx.erdos_renyi_graph(self.N, .2, seed=self.seed)
        while not nx.is_connected(self.graph):
            self.seed += 1
            self.graph = nx.erdos_renyi_graph(self.N, .2, seed=self.seed)
            logging.debug(f"Incrementing seed to {self.seed}.")
        self.pos = nx.spring_layout(self.graph, seed=self.seed)
        for aid in range(self.N):
            citizen = Citizen(aid, self)
            self.schedule.add(citizen)
    def step(self):
        self.schedule.step()
        self.iter += 1
    def display(self):
        nx.draw_networkx(self.graph, pos=self.pos,
            node_color=[ a.opinion for a in self.schedule.agents ],
            ax=self.ax)
        self.ax.set_title(f"Iteration {self.iter} of {self.MAX_STEPS}")
        plt.pause(.1)


parser = argparse.ArgumentParser(description="Marina model.")
parser.add_argument("-n", "--num_sims", type=int, default=1,
    help="Number of sims to run (1 = single, >1 = batch)")
parser.add_argument("-N", type=int, default=15, help="Number of agents.")
parser.add_argument("--MAX_STEPS", type=int, default=50,
    help="Maximum number of steps before simulation terminates.")
parser.add_argument("--seed", type=int, default=138, help="Random seed.")

            

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Parse arguments.
    args = parser.parse_args()

    if args.num_sims == 1:
        soc = Society(**vars(args))

    for s in range(args.MAX_STEPS):
        soc.step()
    input("Press ENTER to close.")
    plt.close(soc.fig)
