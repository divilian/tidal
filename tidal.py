
from mesa import Model, Agent
from mesa.time import RandomActivation
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.schedule = RandomActivation(self)
        for aid in range(self.N):
            citizen = Citizen(aid, self)
            self.schedule.add(citizen)
    def step(self):
        self.schedule.step()
            
 
soc = Society(5)
for step in range(6):
    soc.step()
