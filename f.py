import random
import math
import numpy as np
from tqdm import tqdm as log_progress
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw


class Particle:
    def __init__(self, X, V, r, m):
        self.X = X
        self.V = V
        self.r = r
        self.m = m


class GasSimulation:
    def __init__(self, L, N, m, r, V0, dt, num_steps, create_gif):
        self.L = L
        self.N = N
        self.m = m
        self.r = r
        self.V0 = V0
        self.dt = dt
        self.particles = []
        self.dir_path = 'results'
        self.num_steps = num_steps
        self.create_gif = create_gif

    def initialize_particles(self):
        print('Creating particles')
        existing_positions = set()
        while len(self.particles) < self.N:
            x = random.uniform(self.r * self.V0 * self.dt, self.L - self.r * self.V0 * self.dt)
            y = random.uniform(self.r * self.V0 * self.dt, self.L - self.r * self.V0 * self.dt)
            theta = random.uniform(0, 2 * math.pi)
            vx = self.V0 * math.cos(theta)
            vy = self.V0 * math.sin(theta)
            X = np.array([x, y])
            V = np.array([vx, vy])
            particle = Particle(X, V, self.r, self.m)

            # Check for intersections with existing particles using set
            is_valid = all(
                math.sqrt(np.sum((p.X - particle.X) ** 2)) >= 2 * self.r for p in existing_positions
            )

            if is_valid:
                self.particles.append(particle)
                existing_positions.add(particle)

        print('Particles Created')

    def collisions_particles(self):
        # Create arrays of particle positions and velocities
        positions = np.array([particle.X for particle in self.particles])

        # Calculate distances between all pairs of particles
        distances = np.sqrt(np.sum((positions[:, np.newaxis] - positions) ** 2, axis=-1))

        # Find colliding particles
        colliding_particles = np.argwhere((distances <= 2 * self.r) & (distances > 0))

        # Update velocities of colliding particles
        for (i, j) in colliding_particles:
            particle1 = self.particles[i]
            particle2 = self.particles[j]
            r = (particle1.X - particle2.X) / np.linalg.norm(particle1.X - particle2.X)
            q = -2 * (self.m ** 2 / (2 * self.m)) * (np.dot((particle1.V - particle2.V), r) * r)
            particle1.V += q / self.m
            particle2.V -= q / particle2.m

    def collisions_walls(self):
        # Create arrays of particle positions
        positions = np.array([particle.X for particle in self.particles])

        # Find particles hitting the walls
        hitting_walls = np.argwhere(
            (positions[:, 0] <= self.r) | (positions[:, 0] + self.r >= self.L) |
            (positions[:, 1] - self.r <= 0) | (positions[:, 1] + self.r >= self.L)
        ).flatten()

        # Update velocities of particles hitting walls
        for i in hitting_walls:
            particle = self.particles[i]
            if particle.X[0] <= particle.r or particle.X[0] + particle.r >= self.L:
                particle.V[0] = -particle.V[0]
            if particle.X[1] - particle.r <= 0 or particle.X[1] + particle.r >= self.L:
                particle.V[1] = -particle.V[1]

    def update_position(self):
        for particle in self.particles:
            particle.X += particle.V * self.dt

    def simulate(self):
        self.initialize_particles()
        pbar = log_progress(range(self.num_steps))
        frames = []

        for _ in pbar:
            self.collisions_particles()
            self.collisions_walls()
            self.update_position()

            if self.create_gif:
                frame = self.create_frame_image()
                frames.append(frame)

        if self.create_gif:
            self.save_animation(frames)

        self.plot_velocity()

    def create_frame_image(self):
        scale_factor = 40

        image_width = int(self.L * scale_factor)
        image_height = int(self.L * scale_factor)

        image = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(image)

        for particle in self.particles:
            x1 = int((particle.X[0] - particle.r) * scale_factor)
            y1 = int((particle.X[1] - particle.r) * scale_factor)
            x2 = int((particle.X[0] + particle.r) * scale_factor)
            y2 = int((particle.X[1] + particle.r) * scale_factor)

            draw.ellipse([(x1, y1), (x2, y2)], fill='blue')

        return image

    def save_animation(self, frames):
        gif_path = os.path.join(self.dir_path, 'gas_simulation.gif')
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        print(f"GIF animation saved as '{gif_path}'")

    def plot_velocity(self):
        velocities = [np.linalg.norm(particle.V) for particle in self.particles]

        sns.histplot(velocities, stat='density', label='Simulation', color='#9dbbeb', edgecolor='#e9f2f1')

        plt.savefig(os.path.join(self.dir_path, 'velocity_distribution.png'))


L = 6  # Box volume (L^2)
N = 150  # Number of particles
m = 1  # Particle mass
r = 0.05  # Particle radius
V0 = 10  # Initial velocity
dt = 0.005  # Time step
num_steps = 1000  # Number of simulation steps
create_gif = True  # Set to True to create the GIF animation

simulation = GasSimulation(L, N, m, r, V0, dt, num_steps, create_gif)
simulation.simulate()
