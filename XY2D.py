import numpy as np
import matplotlib.pyplot as plt
import networkx as net
import random
import math

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#want to make my plots look a bit better...
mpl.rcParams['axes.linewidth'] = 1.5 #set the value globally
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
class XY2D:
    def __init__(self, L, beta, seed=None):
        self.beta, self.L = beta, L
        self.rotation_angle = None  # change rotation angle with every step

        ### Initialize random lattice
        self.G = net.grid_graph(dim=(L, L), periodic=True)
        self.initialize(seed)

        self._E, self._mag = 0, 0
        self._theta = np.array(list(net.get_node_attributes(self.G, 'theta').values()))
        self._magnetization = [0, 0]

        self.susceptibility, self.m, self.cluster_size, self.totalE, self.time_averagedE = [], [], [], [], []

        self.steps, self.steps_equilibrium = 0, 0
        self.equilibrium = False

    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        for i in range(self.L):
            for j in range(self.L):
                theta = np.random.uniform(-1*np.pi, np.pi)
                self.G.nodes[i, j]['theta'] = theta
                self.G.nodes[i, j]['s'] = np.array([np.cos(theta), np.sin(theta)])
                self.G.nodes[i, j]['marked'] = False

    def reset(self):
        for i in range(self.L):
            for j in range(self.L):
                self.G.nodes[i, j]['marked'] = False

    def check_equilibrium(self):
        if len(self.time_averagedE) > 20:
            if abs(self.time_averagedE[-20] - self.time_averagedE[-1]) <= 5*self.beta:
                self.steps_equilibrium = self.steps
                self.equilibrium = True
    def record_energy(self):
        if self.steps > 10:
            self.time_averagedE.append(np.mean(self.totalE[-9:]))
        self.totalE.append(self.E)

    def record_magnetization(self):
        self.m.append(self.magnetization)

    def record_susceptibility(self):
        self.susceptibility.append((self.total_magnetization[0] ** 2 + self.total_magnetization[1] ** 2) / (self.L**2))

    def record_cluster_size(self):
        self.cluster_size.append(self.size)

    def rotate(self, spin):
        theta = np.random.uniform(-1*np.pi, np.pi)
        if self.rotation_angle is None:
            self.rotation_angle = np.array([np.cos(theta), np.sin(theta)])
        return spin - 2 * np.dot(spin, self.rotation_angle) * self.rotation_angle

    def step(self):
        self.rotation_angle = None
        idx = random.randint(0, len(self.G) - 1)
        s_i = idx // self.L, idx % self.L

        theta = self.G.nodes[s_i]['theta']
        v = np.array([np.cos(theta), np.sin(theta)])
        new_v = self.rotate(v)

        new_theta = math.acos(new_v[0])
        self.G.nodes[s_i]['theta'] = new_theta
        self.G.nodes[s_i]['s'] = new_v
        self.G.nodes[s_i]['marked'] = True

        self.visit_neighbors(s_i)

        self.record_energy()
        self.record_susceptibility()
        self.record_magnetization()
        self.record_cluster_size()

        self.steps += 1

        if not self.equilibrium:
            self.check_equilibrium()

        self.reset()


    def plot2D(self, fig, axis):
        im = axis.imshow(np.reshape(self.theta, (self.L, self.L)), "hsv", vmin=-1*np.pi, vmax=np.pi)
        fig.colorbar(im)
        # if self.check_equilibrium:
        #     plt.title(f'REACHED EQUILIBRIUM @ {self.steps_equilibrium}')
        return axis
        # plt.title(fr'$\beta$ = {self.beta}')
        # plt.show()

    def show(self, save=None):
        config_matrix = np.reshape(self.theta, (self.L, self.L))
        X, Y = np.meshgrid(np.arange(0, self.L), np.arange(0, self.L))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)
        plt.figure(figsize=(6, 6), dpi=100)
        plt.title(fr'Equilibrium State, $\beta$={self.beta}, {self.steps} steps')
        Q = plt.quiver(X, Y, U, V, units='width')

        plt.axis('off')
        if save is not None:
            plt.savefig(save, bbox_inches='tight')
        plt.show()

    def visit_neighbors(self, si):
        assert self.rotation_angle is not None, 'Please set a rotation angle'
        for neighbor in iter(self.G[si]):
            theta = self.G.nodes[neighbor]['theta']
            vec = np.array([np.cos(self.G.nodes[si]['theta']), np.sin(self.G.nodes[si]['theta'])])
            assert vec.all() == self.G.nodes[neighbor]['s'].all(), 'Something went wrong...'
            if not self.G.nodes[neighbor]['marked']:
                s = np.array([np.cos(theta), np.sin(theta)])
                if random.uniform(0, 1) <= 1 - np.exp(
                        min([0, 2 * self.beta * np.dot(self.rotation_angle, vec) * np.dot(self.rotation_angle, s)])):
                    # print(f'Flipped {neighbor}')
                    new_vec = self.rotate(s)
                    self.G.nodes[neighbor]['theta'] = math.acos(new_vec[0])
                    self.G.nodes[neighbor]['s'] = new_vec
                    self.G.nodes[neighbor]['marked'] = True
                    self.visit_neighbors(neighbor)
        return None

    @property
    def theta(self):
        return np.array(list(net.get_node_attributes(self.G, 'theta').values()))

    @property
    def E(self):
        E = 0
        for node in iter(self.G.nodes):
            theta_i = self.G.nodes[node]['theta']
            for neighbor in iter(self.G[node]):
                theta_j = self.G.nodes[neighbor]['theta']
                E += -np.cos(theta_i - theta_j)
        return E

    @property
    def total_magnetization(self):
        mag = np.array([0.0, 0.0])
        for node in iter(self.G.nodes):
            mag += self.G.nodes[node]['s']
        return mag

    @property
    def size(self):
        S = 0
        for node in iter(self.G.nodes):
            if self.G.nodes[node]['marked']:
                S += 1
        return S

    @property
    def magnetization(self):
        m = []
        for node in iter(self.G.nodes):
            m.append(self.G.nodes[node]['s'])
        return m