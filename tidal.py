
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
            neigh.receive_comm(self)
    def receive_comm(self, neigh):
        logging.info(f"I'm agent {self.unique_id} and I got {neigh.opinion} "\
            f"(with confidence {neigh.confidence:.2f})")
        if self.opinion == neigh.opinion:
            orig_self_conf = self.confidence
            self.confidence += (self.model.confidence_malleability *
                neigh.confidence)
            if self.confidence > 1 and self.model.cap_confidence:
                self.confidence = 1
            if self.model.bidirectional_influence:
                neigh.confidence += (self.model.confidence_malleability *
                    orig_self_conf)
                if neigh.confidence > 1 and self.model.cap_confidence:
                    neigh.confidence = 1
        else:
            orig_self_conf = self.confidence
            self.confidence -= (self.model.confidence_malleability *
                neigh.confidence)
            if self.confidence <= 0:
                # Okay, I give!
                self.opinion = neigh.opinion
                self.confidence = self.base_confidence
            if self.model.bidirectional_influence:
                neigh.confidence -= (self.model.confidence_malleability *
                    orig_self_conf)
                if neigh.confidence <= 0:
                    # Okay, I give!
                    neigh.opinion = self.opinion
                    neigh.confidence = neigh.base_confidence
        if not self.model.animate_only_on_step:
            self.model.display()


class Society(Model):
    def __init__(self, **kwd_args):
        super().__init__()
        for arg, val in kwd_args.items():
            setattr(self, arg, val)
        self.rng = np.random.RandomState(seed=self.seed)
        self.schedule = BaseScheduler(self)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        self.fig.suptitle(f"Iteration 0 of {self.MAX_STEPS}")
        self.fig.tight_layout()
        self.iter = 0
        self.graph = self.gen_social_network()
        while not nx.is_connected(self.graph):
            self.seed += 1
            logging.debug(f"Incrementing seed to {self.seed}.")
            self.graph = self.gen_social_network()
        self.pos = nx.spring_layout(self.graph, seed=self.seed)
        for aid in range(self.N):
            citizen = Citizen(aid, self)
            self.schedule.add(citizen)
    def gen_social_network(self):
        return nx.erdos_renyi_graph(self.N, .12, seed=self.seed)
    def step(self):
        self.schedule.step()
        self.iter += 1
        if self.animate_only_on_step:
            self.display()
    def display(self):
        self.ax[0].cla()
        self.ax[1].cla()
        nx.draw_networkx(self.graph, pos=self.pos,
            node_color=[ a.opinion for a in self.schedule.agents ],
            node_size=[ a.confidence * 300 + 50 for a in self.schedule.agents ],
            ax=self.ax[0])
        self.fig.suptitle(f"Iteration {self.iter} of {self.MAX_STEPS}")
        confs = np.array([ a.confidence if a.opinion == "red" else -a.confidence
            for a in self.schedule.agents ])
        confs_blue = confs[confs < 0]
        confs_gray = confs[confs == 0]
        confs_red = confs[confs >= 0]
        if self.cap_confidence:
            bins = np.linspace(-1, 1, 51)
        else:
            max_abs = max(-confs.min(), confs.max())
            bins = np.linspace(-max_abs, max_abs, 51)
        self.ax[1].hist([confs_blue, confs_gray, confs_red], bins=bins,
            color=["blue","gray","red"], width=(bins[1]-bins[0]),
            edgecolor="black")
        self.ax[1].set_title("Confidence levels")
        if self.cap_confidence:
            self.ax[1].set_xlim((-1,1.05))
        self.ax[1].set_ylim((0, self.N))
        if self.plot_mean:
            the_mean = confs.mean()
            the_median = np.median(confs)
            self.ax[1].axvline(x=the_mean,
                color= "red" if the_mean > 0 else "blue",
                linestyle="dashed")
            self.ax[1].axvline(x=the_median,
                color= "red" if the_mean > 0 else "blue", alpha=.2,
                linestyle="dashed")
            self.ax[1].text(the_mean + .02, self.N * .9, "mean",
                color= "red" if the_mean > 0 else "blue",
                rotation=90)
            self.ax[1].text(the_median + .02, self.N * .75, "median", alpha=.2,
                color= "red" if the_mean > 0 else "blue",
                rotation=90)
        plt.pause(.1)


parser = argparse.ArgumentParser(description="Tidal model.")
parser.add_argument("-n", "--num_sims", type=int, default=1,
    help="Number of sims to run (1 = single, >1 = batch)")
parser.add_argument("-N", type=int, default=15, help="Number of agents.")
parser.add_argument("--MAX_STEPS", type=int, default=50,
    help="Maximum number of steps before simulation terminates.")
parser.add_argument("--seed", type=int, default=138, help="Random seed.")
parser.add_argument("--confidence_malleability", type=float, default=1/3.,
    help="To what degree is confidence boosted/devastated by (dis)agreement.")
parser.add_argument("--cap_confidence", action=argparse.BooleanOptionalAction,
    help="Impose maximum confidence of 1.0 on each agent?")
parser.add_argument("--animate_only_on_step",
    action=argparse.BooleanOptionalAction,
    help="Only draw new animation frame on entire step of sim?")
parser.add_argument("--plot_mean", action=argparse.BooleanOptionalAction,
    help="Plot mean and median confidence on histogram?")
parser.add_argument("--bidirectional_influence",
    action=argparse.BooleanOptionalAction,
    help="Listener also influences speaker (symmetrically)?")

            

if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING)

    # Parse arguments.
    args = parser.parse_args()

    if args.num_sims == 1:
        soc = Society(**vars(args))

    for s in range(args.MAX_STEPS):
        soc.step()
    input("Press ENTER to close.")
    plt.close(soc.fig)
