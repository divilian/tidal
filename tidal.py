
from mesa import Model, Agent
from mesa.time import RandomActivation
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        print(f"Hi, I'm agent {self.unique_id}.")



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
