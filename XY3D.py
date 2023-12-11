import numpy as np
import matplotlib.pyplot as plt
import networkx as net
import random
import math

class XY3D:
    def __init__(self, L, beta, seed=None):
        self.beta, self.L = beta, L
        self.rotation_angle = None  # change rotation angle with every step

        ### Initialize random lattice
        self.G = net.grid_graph(dim=(L, L, L), periodic=True)
        self.initialize(seed)

        self._E, self._mag = 0, 0
        self._theta = np.array(list(net.get_node_attributes(self.G, 'theta').values()))
        self._phi = np.array(list(net.get_node_attributes(self.G, 'phi').values()))
        self._magnetization = [0, 0, 0]

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
                for k in range(self.L):
                    theta = np.random.uniform(np.pi, 2*np.pi)
                    phi = np.random.uniform(0, np.pi)
                    self.G.nodes[i, j, k]['theta'] = theta
                    self.G.nodes[i, j, k]['phi'] = phi
                    self.G.nodes[i, j, k]['s'] = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
                    self.G.nodes[i, j, k]['marked'] = False

    def reset(self):
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    self.G.nodes[i, j, k]['marked'] = False

    def record_energy(self):
        self.totalE.append(self.E)

    def record_magnetization(self):
        self.m.append(self.mag)

    def record_susceptibility(self):
        self.susceptibility.append((self.magnetization[0]**2 + self.magnetization[1]**2 + self.magnetization[2]**2) / (self.L**3))

    def record_cluster_size(self):
        self.cluster_size.append(self.size)

    def rotate(self, spin):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        if self.rotation_angle is None:
            self.rotation_angle = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        return spin - 2 * np.dot(spin, self.rotation_angle) * self.rotation_angle

    def step(self):
        self.rotation_angle = None
        idx = random.randint(0, len(self.G) - 1)
        s_i = idx // (self.L ** 2), (idx // self.L) % self.L, idx % self.L

        theta = self.G.nodes[s_i]['theta']
        phi = self.G.nodes[s_i]['phi']
        v = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        new_v = self.rotate(v)
        new_theta = math.atan2(new_v[1], new_v[0])
        new_phi = math.acos(new_v[2])

        if ((new_v - np.array([np.sin(new_phi) * np.cos(new_theta), np.sin(new_phi) * np.sin(new_theta), np.cos(new_phi)])) > 1e-10).any():
            print('Error')
            print((new_v - np.array([np.sin(new_phi) * np.cos(new_theta), np.sin(new_phi) * np.sin(new_theta), np.cos(new_phi)])))

        self.G.nodes[s_i]['theta'] = new_theta
        self.G.nodes[s_i]['phi'] = new_phi
        self.G.nodes[s_i]['s'] = new_v
        self.G.nodes[s_i]['marked'] = True

        self.visit_neighbors(s_i)

        self.record_energy()
        self.record_susceptibility()
        self.record_magnetization()

        self.steps += 1
        self.reset()

    def plot2D(self, fig, axis, which='theta'):
        if which=='theta':
            im = axis.imshow(np.reshape(self.theta, (self.L, self.L, self.L))[:, :, self.L // 2], "hsv", vmin=-1*np.pi, vmax=np.pi)
        elif which=='phi':
            im = axis.imshow(np.reshape(self.phi, (self.L, self.L, self.L))[:, :, self.L // 2], "hsv", vmin=0, vmax=np.pi)
        fig.colorbar(im)
        return axis

    def show(self, save=None, var='theta'):
        if var == 'theta':
            config_matrix = np.reshape(self.theta, (self.L, self.L, self.L))[:, :, self.L // 2]
        else:
            raise ValueError

        X, Y = np.meshgrid(np.arange(0, self.L), np.arange(0, self.L))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)

        plt.figure(figsize=(6, 6), dpi=100)
        #plt.title(fr'Equilibrium State, $\beta$={self.beta}, {self.steps} steps')
        Q = plt.quiver(X, Y, U, V, units='width')

        plt.axis('off')
        if save is not None:
            plt.savefig(save, bbox_inches='tight')
        plt.show()

    def visit_neighbors(self, si):
        assert self.rotation_angle is not None, 'Please set a rotation angle'
        for neighbor in iter(self.G[si]):
            theta = self.G.nodes[neighbor]['theta']
            phi = self.G.nodes[neighbor]['phi']
            vec = np.array([np.sin(self.G.nodes[si]['phi']) * np.cos(self.G.nodes[si]['theta']), np.sin(self.G.nodes[si]['phi']) * np.sin(self.G.nodes[si]['theta']), np.cos(self.G.nodes[si]['phi'])])
            if not self.G.nodes[neighbor]['marked']:
                s = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
                if random.uniform(0, 1) <= 1 - np.exp(
                        min([0, 2 * self.beta * np.dot(self.rotation_angle, vec) * np.dot(self.rotation_angle, s)])):

                    new_vec = self.rotate(s)

                    self.G.nodes[neighbor]['theta'] = math.atan2(new_vec[1], new_vec[0])
                    self.G.nodes[neighbor]['phi'] = math.acos(new_vec[2])
                    self.G.nodes[neighbor]['s'] = new_vec
                    self.G.nodes[neighbor]['marked'] = True
                    self.visit_neighbors(neighbor)
        return None

    @property
    def theta(self):
        return np.array(list(net.get_node_attributes(self.G, 'theta').values()))

    @property
    def phi(self):
        return np.array(list(net.get_node_attributes(self.G, 'phi').values()))

    @property
    def E(self):
        E = 0
        for node in iter(self.G.nodes):
            s1 = self.G.nodes[node]['s']
            for neighbor in iter(self.G[node]):
                s2 = self.G.nodes[neighbor]['s']
                E += -1*np.dot(s1, s2)
        return E

    @property
    def magnetization(self):
        mag = np.array([0.0, 0.0, 0.0])
        for node in iter(self.G.nodes):
            mag += self.G.nodes[node]['s']
        return mag

    @property
    def mag(self):
        m = []
        for node in iter(self.G.nodes):
            m.append(self.G.nodes[node]['s'])
        return m