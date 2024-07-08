
from mesa import Model, Agent
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import logging
from collections import Counter



class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = self.model.rng.choice(["red","blue"],
            p=[model.prop_red, 1-model.prop_red])
        self.base_confidence = self.model.rng.uniform(0,1,1)[0]
        self.confidence = self.base_confidence
        self.interactions_with_alike = 0
        self.interactions_with_diff = 0
        self.convs_to = {'red':0, 'blue':0}
    def step(self):
        if not self.model.animate_only_on_step:
            self.model.display()
    def reinforce_opinion(self, reinf_conf, bidirectional=False):
        orig_self_conf = self.confidence
        self.interactions_with_alike += 1
        self.confidence += (self.model.confidence_malleability * reinf_conf)
        if self.confidence > 1 and self.model.cap_confidence:
            self.confidence = 1
        if bidirectional:
            neigh.reinforce_opinion(orig_self_conf, False)
    def challenge_opinion(self, challenge_conf, challenge_op,
            bidirectional=False):
        self.interactions_with_diff += 1
        orig_self_conf = self.confidence
        self.confidence -= (self.model.confidence_malleability * challenge_conf)
        if self.confidence <= 0:
            # Okay, I give!
            self.record_conversion(challenge_op)
            self.opinion = challenge_op
            self.confidence = self.base_confidence
        if bidirectional:
            neigh.challenge_opinion(orig_self_conf, False)
    def balance_opinion(self, challenge_conf, balanced_op):
        self.confidence += (self.model.confidence_malleability * challenge_conf)
        if self.confidence > 1 and self.model.cap_confidence:
            self.confidence = 1
        if self.opinion != balanced_op:
            self.record_conversion(balanced_op)
        self.opinion = balanced_op
        self.confidence = self.base_confidence
    def record_conversion(self, new_opinion):
        self.convs_to[new_opinion] += 1


# A messaging citizen influences one agent at a time, to a random one of its
# graph neighbors.
class MessagingCitizen(Citizen):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        logging.info(f"Hi, I'm messaging agent {self.unique_id}.")
        if self.model.rng.uniform(0,1,1)[0] < self.model.extraversion:
            neighnums = list(self.model.graph.neighbors(self.unique_id))
            neigh = self.model.schedule.agents[self.model.rng.choice(neighnums)]
            logging.info(f" ..and I'm talking at agent {neigh.unique_id}.")
            if self.opinion == neigh.opinion:
                self.reinforce_opinion(neigh.confidence)
            else:
                self.challenge_opinion(neigh.confidence, neigh.opinion)
        super().step()



# A community citizen receives influence all at once, from all its graph
# neighbors.
class CommunityCitizen(Citizen):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.peacemaker = self.model.rng.choice([True,False],
            p=[self.model.prop_peacemakers, 1-self.model.prop_peacemakers])
        self.pm_convs_to = {'red':0, 'blue':0}
    def step(self):
        logging.info(f"Hi, I'm community agent {self.unique_id} "
            f"{'(peacemaker)' if self.peacemaker else ''}.")
        # Note: "extraversion" here means propensity to be influenced, not to
        # influence.
        if self.model.rng.uniform(0,1,1)[0] < self.model.extraversion:
            neighnums = list(self.model.graph.neighbors(self.unique_id))
            logging.info(f" ..and I'm listening to {neighnums}.")
            ops = [ self.model.schedule.agents[n].opinion for n in neighnums ]
            op_ctr = Counter(ops)
            # (This will all break if more than 2 opinions.)
            if (len(op_ctr.most_common(2)) == 2 and
                op_ctr.most_common(2)[0][0] == op_ctr.most_common(2)[0][1]):
                # (Necessary to avoid privileging red or blue in case of ties.)
                logging.info(f" ..choosing desired_op randomly.")
                desired_op = self.model.rng.choice(["red","blue"])
            elif self.peacemaker:
                desired_op = ("red" if op_ctr.most_common(1)[0][0] == "blue"
                                                                else "blue")
            else:
                desired_op = op_ctr.most_common(1)[0][0]
            logging.info(f" ..desired opinion: {desired_op}")
            community_confidence = op_ctr[desired_op] / len(ops)
            if self.opinion == desired_op:
                self.reinforce_opinion(community_confidence)
            else:
                if self.peacemaker:
                    self.balance_opinion(community_confidence, desired_op)
                else:
                    self.challenge_opinion(community_confidence, desired_op)
    def record_conversion(self, new_opinion):
        super().record_conversion(new_opinion)
        if self.peacemaker:
            self.pm_convs_to[new_opinion] += 1


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
            citizen = globals()[self.agent_class + "Citizen"](aid, self)
            self.schedule.add(citizen)
        self.datacollector = DataCollector(
            model_reporters={'alikes':Society.num_alikes,
                'diffs':Society.num_diffs,
                'convs_to_red':Society.num_conversions_to_red,
                'convs_to_blue':Society.num_conversions_to_blue,
                'peacemaker_convs_to_red':Society.num_pm_conversions_to_red,
                'peacemaker_convs_to_blue':Society.num_pm_conversions_to_blue}
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
    def num_pm_conversions_to_red(self):
        if self.agent_class == 'Community':
            return self.num_conversions_to('red',True)
        else:
            return 0
    def num_pm_conversions_to_blue(self):
        if self.agent_class == 'Community':
            return self.num_conversions_to('blue',True)
        else:
            return 0
    def num_conversions_to(self, color, pm=False):
        return sum([ getattr(a, 'pm_convs_to' if pm else 'convs_to')[color]
                                            for a in self.schedule.agents ])
    def gen_social_network(self):
        if self.graph_type == 'ER':
            if self.graph_params is not None:
                return nx.erdos_renyi_graph(self.N,
                    float(self.graph_params[0]), seed=self.seed)
            else:
                return nx.erdos_renyi_graph(self.N, .1, seed=self.seed)
        else:
            if self.graph_params is not None:
                n1 = int(self.graph_params[0])
                n2 = int(self.graph_params[1])
                p11 = float(self.graph_params[2])
                p12 = float(self.graph_params[3])
                p22 = float(self.graph_params[4])
            else:
                n1 = self.N // 2
                n2 = self.N // 2
                p11 = p22 = .25
                p12 = .01
            return nx.stochastic_block_model([ n1, n2 ],
                [[p11, p12], [p12, p22]], seed=self.seed)
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
            edgecolors=[ "lightgreen" if 'peacemaker' in vars(a) and a.peacemaker else "black"
                for a in self.schedule.agents ],
            linewidths=2.5,
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
            {'alikes':['green','solid'],'diffs':['orange','solid']},
            initMax=self.N * self.extraversion)
    def display_conversions(self):
        if self.agent_class == 'Community':
            self.display_time_plot(self.ax[0][0], "Conversions (to color)",
                {'convs_to_red':['red','solid'],
                'convs_to_blue':['blue','solid'],
                'peacemaker_convs_to_red':['red','dashed'],
                'peacemaker_convs_to_blue':['blue','dashed']},
                cumu=True, initMax=self.N * self.extraversion)
        else:
            self.display_time_plot(self.ax[0][0], "Conversions (to color)",
                {'convs_to_red':['red','solid'],
                'convs_to_blue':['blue','solid']},
                cumu=True, initMax=self.N * self.extraversion)
    def display_time_plot(self, axes, title, varsStyles,
        cumu=False, initMax=0):
        axes.cla()
        fudge_factor_initMax = 1.2
        # Compute pairwise differences of this DF, which gives culumative sums.
        df = self.datacollector.get_model_vars_dataframe()
        if not cumu:
            df = df.diff().fillna(0)
        for var in varsStyles.keys():
            axes.plot(df[var], label="cumu_" + var if cumu else var,
                color=varsStyles[var][0],
                linestyle=varsStyles[var][1])
        axes.set_xlim((0,self.MAX_STEPS))
        axes.set_ylim((0,max(df[varsStyles.keys()].max().max(), initMax)))
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
parser.add_argument("--graph_type", choices=['ER','SBM'],
    default='ER', help="Random graph-generating algorithm.")
parser.add_argument("--graph_params", nargs='+',
    help="Random graph-generating algorithm parameters (ER: p. "
        "SBM: n1 n2 p11 p12 p22).")

parser.add_argument("--agent_class", choices=['Messaging','Community'],
    default='Messaging',
    help="Prefix of agent class name (prepended to 'Citizen').")
parser.add_argument("--prop_peacemakers", type=float, default=.1,
    help="Proportion of peacemaker (heterophily-loving) agents "
        "(Community only).")

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
