
from mesa import Model, Agent
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import logging



class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = self.model.rng.choice(["red","blue"],
            p=[model.prop_red, 1-model.prop_red])
        self.base_confidence = self.model.rng.uniform(0,1,1)[0]
        self.confidence = self.base_confidence
        self.interactions_with_alike = 0
        self.interactions_with_diff = 0
        self.conversions_to = {'red':0, 'blue':0}
    def step(self):
        logging.info(f"Hi, I'm agent {self.unique_id}.")
        if self.model.rng.uniform(0,1,1)[0] < self.model.extraversion:
            neighnums = list(self.model.graph.neighbors(self.unique_id))
            neigh = self.model.schedule.agents[self.model.rng.choice(neighnums)]
            logging.info(f" ..and I'm talking at agent {neigh.unique_id}.")
            neigh.receive_comm(self)
    def receive_comm(self, neigh):
        logging.info(f"I'm agent {self.unique_id} and I got {neigh.opinion} "\
            f"(with confidence {neigh.confidence:.2f})")
        if self.opinion == neigh.opinion:
            orig_self_conf = self.confidence
            self.interactions_with_alike += 1
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
            self.interactions_with_diff += 1
            orig_self_conf = self.confidence
            self.confidence -= (self.model.confidence_malleability *
                neigh.confidence)
            if self.confidence <= 0:
                # Okay, I give!
                self.conversions_to[neigh.opinion] += 1
                self.opinion = neigh.opinion
                self.confidence = self.base_confidence
            if self.model.bidirectional_influence:
                neigh.confidence -= (self.model.confidence_malleability *
                    orig_self_conf)
                if neigh.confidence <= 0:
                    # Okay, I give!
                    neigh.conversions_to[self.opinion] += 1
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
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12,9))
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
        self.datacollector = DataCollector(
            model_reporters={'alikes':Society.num_alikes,
                'diffs':Society.num_diffs,
                'convs_to_red':Society.num_conversions_to_red,
                'convs_to_blue':Society.num_conversions_to_blue}
        )
    def num_alikes(self):
        return self.sum_agent_vals('interactions_with_alike')
    def num_diffs(self):
        return self.sum_agent_vals('interactions_with_diff')
    def sum_agent_vals(self, agent_param_name):
        return sum([ getattr(a, agent_param_name)
                                            for a in self.schedule.agents ])
    def num_conversions_to_red(self):
        return self.num_conversions_to('red')
    def num_conversions_to_blue(self):
        return self.num_conversions_to('blue')
    def num_conversions_to(self, color):
        return sum([ getattr(a, 'conversions_to')[color]
                                            for a in self.schedule.agents ])
    def gen_social_network(self):
        return nx.erdos_renyi_graph(self.N, .12, seed=self.seed)
    def step(self):
        self.schedule.step()
        self.iter += 1
        self.datacollector.collect(self)
        if self.animate_only_on_step:
            self.display()
    def display(self):
        self.display_graph()
        self.display_confidences()
        self.display_interactions()
        self.display_conversions()
        self.fig.suptitle(f"Iteration {self.iter} of {self.MAX_STEPS}")
        plt.pause(.1)
    def display_graph(self):
        axes = self.ax[1][0]
        axes.cla()
        nx.draw_networkx(self.graph, pos=self.pos,
            node_color=[ a.opinion for a in self.schedule.agents ],
            node_size=[ a.confidence * 300 + 50 for a in self.schedule.agents ],
            ax=axes)
    def display_confidences(self):
        axes = self.ax[1][1]
        axes.cla()
        confs = np.array(
            [ a.confidence if a.opinion == "red" else -a.confidence
                                            for a in self.schedule.agents ])
        confs_blue = confs[confs < 0]
        confs_gray = confs[confs == 0]
        confs_red = confs[confs >= 0]
        if self.cap_confidence:
            bins = np.linspace(-1, 1, 51)
        else:
            max_abs = max(-confs.min(), confs.max())
            bins = np.linspace(-max_abs, max_abs, 51)
        axes.hist([confs_blue, confs_gray, confs_red], bins=bins,
            color=["blue","gray","red"], width=(bins[1]-bins[0]),
            edgecolor="black")
        axes.set_title("Confidence levels")
        if self.cap_confidence:
            axes.set_xlim((-1,1.05))
        axes.set_ylim((0, self.N))
        if self.plot_mean:
            the_mean = confs.mean()
            the_median = np.median(confs)
            axes.axvline(x=the_mean,
                color= "red" if the_mean > 0 else "blue",
                linestyle="dashed")
            axes.axvline(x=the_median,
                color= "red" if the_mean > 0 else "blue", alpha=.2,
                linestyle="dashed")
            axes.text(the_mean + .02, self.N * .9, "mean",
                color= "red" if the_mean > 0 else "blue",
                rotation=90)
            axes.text(the_median + .02, self.N * .75, "median", alpha=.2,
                color= "red" if the_mean > 0 else "blue",
                rotation=90)
    def display_interactions(self):
        self.display_time_plot(self.ax[0][1], "Interactions (by likeness)",
            {'alikes':'green','diffs':'orange'},
            initMax=self.N * self.extraversion)
    def display_conversions(self):
        self.display_time_plot(self.ax[0][0], "Conversions (to color)",
            {'convs_to_red':'red','convs_to_blue':'blue'}, cumu=True,
            initMax=self.N * self.extraversion)
    def display_time_plot(self, axes, title, varsColors, cumu=False, initMax=0):
        axes.cla()
        fudge_factor_initMax = 1.2
        # Compute pairwise differences of this DF, which gives culumative sums.
        df = self.datacollector.get_model_vars_dataframe()
        if not cumu:
            df = df.diff().fillna(0)
        for var in varsColors.keys():
            axes.plot(df[var], label="cumu_" + var if cumu else var,
            color=varsColors[var])
        axes.set_xlim((0,self.MAX_STEPS))
        axes.set_ylim((0,max(df[varsColors.keys()].max().max(), initMax)))
        axes.set_title("Cumulative " + title.lower() if cumu else title)
        axes.set_xlabel("Iteration")
        axes.legend()


# Simulation parameters.
parser = argparse.ArgumentParser(description="Tidal model.")
parser.add_argument("-n", "--num_sims", type=int, default=1,
    help="Number of sims to run (1 = single, >1 = batch)")
parser.add_argument("--seed", type=int, default=138, help="Random seed.")
parser.add_argument("--MAX_STEPS", type=int, default=50,
    help="Maximum number of steps before simulation terminates.")

parser.add_argument("-N", type=int, default=15, help="Number of agents.")

parser.add_argument("--prop_red", type=float, default=.5,
    help="Proportion of agents initially red.")
parser.add_argument("--confidence_malleability", type=float, default=1/3.,
    help="To what degree is confidence boosted/devastated by (dis)agreement.")
parser.add_argument("--cap_confidence", action=argparse.BooleanOptionalAction,
    help="Impose maximum confidence of 1.0 on each agent?")
parser.add_argument("--bidirectional_influence",
    action=argparse.BooleanOptionalAction,
    help="Listener also (symmetrically) influences speaker?")
parser.add_argument("--extraversion", type=float, default=.7,
    help="Probability of each agent sending message each step.")

parser.add_argument("--animate_only_on_step",
    action=argparse.BooleanOptionalAction,
    help="Only draw new animation frame on entire step of sim?")
parser.add_argument("--plot_mean", action=argparse.BooleanOptionalAction,
    help="Plot mean and median confidence on histogram?")

            

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
