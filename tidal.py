
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
        self.advocate = self.model.rng.choice([True,False],
            p=[self.model.prop_advocates, 1-self.model.prop_advocates])
        self.adv_convs_to = {'red':0, 'blue':0}
    def step(self):
        logging.info(f"Hi, I'm community agent {self.unique_id} "
            f"{'(advocate)' if self.advocate else ''}.")
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
            elif self.advocate:
                desired_op = ("red" if op_ctr.most_common(1)[0][0] == "blue"
                                                                else "blue")
            else:
                desired_op = op_ctr.most_common(1)[0][0]
            logging.info(f" ..desired opinion: {desired_op}")
            # TODO: should confidence of neighbors come into play?
            community_confidence = op_ctr[desired_op] / len(ops)
            if self.opinion == desired_op:
                self.reinforce_opinion(community_confidence)
            else:
                if self.advocate:
                    self.balance_opinion(community_confidence, desired_op)
                else:
                    self.challenge_opinion(community_confidence, desired_op)
    def record_conversion(self, new_opinion):
        super().record_conversion(new_opinion)
        if self.advocate:
            self.adv_convs_to[new_opinion] += 1


class Society(Model):
    def __init__(self, **kwd_args):
        super().__init__()
        for arg, val in kwd_args.items():
            setattr(self, arg, val)
        self.rng = np.random.RandomState(seed=self.seed)
        self.schedule = BaseScheduler(self)
        if self.num_sims == 1:
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
            model_reporters={
                'alikes':Society.num_alikes,
                'diffs':Society.num_diffs,
                'prop_blue':Society.prop_blue,
                'mean_conf_reg':Society.mean_conf_reg,
                'mean_conf_adv':Society.mean_conf_adv,
                'convs_to_red':Society.num_conversions_to_red,
                'convs_to_blue':Society.num_conversions_to_blue,
                'advocate_convs_to_red':Society.num_adv_conversions_to_red,
                'advocate_convs_to_blue':Society.num_adv_conversions_to_blue}
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
    def prop_blue(self):
        return np.array(
            [ a.opinion == "blue" for a in self.schedule.agents ]).mean()
    def mean_conf_reg(self):
        return self.mean_conf(False)
    def mean_conf_adv(self):
        return self.mean_conf(True)
    def mean_conf(self, adv=False):
        nodes = [ a for a in self.schedule.agents
            if 'advocate' not in vars(a)  or  a.advocate == adv ]
        return sum([ abs(n.confidence) for n in nodes ]) / len(nodes)
    def num_conversions_to_blue(self):
        return self.num_conversions_to('blue')
    def num_adv_conversions_to_red(self):
        if self.agent_class == 'Community':
            return self.num_conversions_to('red',True)
        else:
            return 0
    def num_adv_conversions_to_blue(self):
        if self.agent_class == 'Community':
            return self.num_conversions_to('blue',True)
        else:
            return 0
    def num_conversions_to(self, color, adv=False):
        return sum([ getattr(a, 'adv_convs_to' if adv else 'convs_to')[color]
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
        if self.num_sims == 1:
            self.display_graph(self.ax[1][0])
            self.display_confidence_hist(self.ax[1][1])
            self.display_confidence_polarity(self.ax[0][1])
            self.display_conversions(self.ax[0][0])
            self.fig.suptitle(f"Iteration {self.iter} of {self.MAX_STEPS}")
            plt.pause(.1)
    def display_graph(self, axes):
        axes.cla()
        nx.draw_networkx(self.graph, pos=self.pos,
            node_color=[ a.opinion for a in self.schedule.agents ],
            node_size=[ a.confidence * 300 + 40 for a in self.schedule.agents ],
            node_shape='o',
            ax=axes)
        adv_nodes = { a.unique_id for a in self.schedule.agents
            if 'advocate' in vars(a) and a.advocate }
        adv_graph = self.graph.subgraph(adv_nodes)
        nx.draw_networkx(adv_graph, pos=self.pos,
            node_color=[ a.opinion for a in self.schedule.agents
                                                if a.unique_id in adv_nodes ],
            node_size=[ a.confidence * 450 + 80 for a in self.schedule.agents
                                                if a.unique_id in adv_nodes  ],
            node_shape='s',
            edgecolors='black',
            linewidths=3,
            ax=axes)
    def display_confidence_hist(self, axes):
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
    def display_confidence_polarity(self, axes):
        self.display_time_plot(axes, "Polarity/confidence",
            {'prop_blue':['blue','solid'],
            'mean_conf_adv':['green','dashed'],
            'mean_conf_reg':['black','dotted']}, True, 1.0)
    def display_interactions(self, axes):
        self.display_time_plot(axes, "Interactions (by likeness)",
            {'alikes':['green','solid'],'diffs':['orange','solid']},
            initMax=self.N * self.extraversion)
    def display_conversions(self, axes):
        if self.agent_class == 'Community':
            self.display_time_plot(axes, "Cumulative conversions (to color)",
                {'convs_to_red':['red','solid'],
                'convs_to_blue':['blue','solid'],
                'advocate_convs_to_red':['red','dashed'],
                'advocate_convs_to_blue':['blue','dashed']},
                cumu=True, initMax=self.N * self.extraversion)
        else:
            self.display_time_plot(axes, "Cumulative conversions (to color)",
                {'convs_to_red':['red','solid'],
                'convs_to_blue':['blue','solid']},
                cumu=True, initMax=self.N * self.extraversion)
    def display_time_plot(self, axes, title, varsStyles,
        cumu=False, initMax=0):
        axes.cla()
        # Compute pairwise differences of this DF, which gives culumative sums.
        df = self.datacollector.get_model_vars_dataframe()
        if not cumu:
            df = df.diff().fillna(0)
        for var in varsStyles.keys():
            axes.plot(df[var], label=var,
                color=varsStyles[var][0],
                linestyle=varsStyles[var][1])
        axes.set_xlim((0,self.MAX_STEPS))
        axes.set_ylim((0,max(df[varsStyles.keys()].max().max(), initMax)))
        axes.set_title(title)
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
parser.add_argument("--prop_advocates", type=float, default=.1,
    help="Proportion of advocate (heterophily-loving) agents "
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

        # Individual run.
        soc = Society(**vars(args))

        for s in range(args.MAX_STEPS):
            soc.step()
        input("Press ENTER to close.")
        plt.close(soc.fig)

    else:

        # Batch run.
        pass
