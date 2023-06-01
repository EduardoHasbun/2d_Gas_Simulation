import random
import math
from PIL import Image, ImageDraw
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np

class Particle:
    def __init__(self, X, V, r, m):
        self.X = X
        self.V = V
        self.r = r
        self.m = m

class GasSimulation:
    def __init__(self, L, N, m, r, V0, dt):
        self.L = L
        self.N = N
        self.m = m
        self.r = r
        self.V0 = V0
        self.dt = dt
        self.particles = []
        self.momentum = 0
        self.wall_collisions = 0

    def initialize_particles(self):
        while len(self.particles) < self.N:
            x = random.uniform(self.r, self.L - self.r)
            y = random.uniform(self.r, self.L - self.r)
            theta = random.uniform(0, 2 * math.pi)
            vx = self.V0 * math.cos(theta)
            vy = self.V0 * math.sin(theta)
            X = np.array([x,y])
            V = np.array([vx,vy])
            particle = Particle(X, V, self.r, self.m)

            # Check for intersections with existing particles
            is_valid = all(math.sqrt((p.X[0] - particle.X[0])**2 + (p.X[1] - particle.X[1])**2) >= 2 * self.r for p in self.particles)

            if is_valid:
                self.particles.append(particle)

    def movement(self):
        for particle in self.particles:
            particle.X += particle.V*self.dt
            
    def calculate_pressure(self):
        pressure = self.momentum/(self.dt*self.L*self.N)
        return pressure
    
    
    def calculate_temperature(self):
        E = 0
        for i in range (self.N):
            particle = self.particles[i]
            E += 0.5*self.m*np.linalg.norm((particle.V))**2
        T = (2/3)*E/(self.N*1.380649*10**-23)-273.15
        return T
    
    
    def momentum(self):
        return self.momentum
    
    def ideal_gas(self,T,P):
        #PV = NKbT
        lhs = P*self.L**2
        rhs = self.N*1.380649*10**-23*T
        total = lhs-rhs
        return total
    

    def collisions(self):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                particle1 = self.particles[i]
                particle2 = self.particles[j]
                distance = np.linalg.norm((particle1.X,particle2.X))
                if distance <= 2 * self.r:  # Collision between particles
                    V1,X1 = particle1.V,particle1.Po
                    V2,X2 = particle2.V,particle2.Po
                    r = (X1-X2)/(np.linalg.norm(X1-X2))
                    q = -2*(self.m**2/(2*self.m))*(np.dot((V1-V2),r)*r)
                    
                    particle1.V += q/self.m
                    particle2.V -= q/particle2.m

            # Check collisions with walls
            particle = self.particles[i]
            if particle.X[0] <= particle.r or particle.X[0] + particle.r >= self.L:
                particle.V[0] = -particle.V[0]
                self.wall_collisions += 1
                self.momentum += 2*particle.m*abs(particle.V[0])
            if particle.X[1] - particle.r <= 0 or particle.X[1] + particle.r >= self.L:
                particle.V[1] = -particle.V[1]
                self.wall_collisions += 1
                self.momentum += 2*particle.m*abs(particle.V[1])
                


    def simulate(self, num_steps):
        self.initialize_particles()
        

        frames = []

        for _ in range(num_steps):
            # Create a new frame for each step
            frame = self.create_frame_image()
            frames.append(frame)

            self.movement()
            self.collisions()
            

        self.save_animation(frames)
        
        #Obtain final velocities
        velocities = [np.linalg.norm(p.V) for p in self.particles]
        pressure = self.calculate_pressure()
        temperature = self.calculate_temperature()
        momentum = self.momentum
        ideal_gas = self.ideal_gas(temperature,pressure)
        
        print(f"Momentum:{momentum}")
        print(f"Pressure:{pressure}")
        print(f"Temperature:{temperature}")
        print(f"Ideal gas:{ideal_gas}")
        
        
        # sns.kdeplot(velocities)
        # # plt.hist(velocities,bins=25, density=True, alpha=0.6, color='g',edgecolor='k',label='Histgram')
        
        # plt.xlabel('Velocity')
        # plt.ylabel('Density')
        # plt.title('Velocity Distribution')
        # plt.show()
    

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
        gif_path = 'gas_simulation.gif'
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        print(f"GIF animation saved as '{gif_path}'")
        

L = 10  # Box volume (L^2)
N = 100  # Number of particles
m = 1  # Particle mass
r = 0.05  # Particle radius
V0 = 10  # Initial velocity
dt = 0.005  # Time step
num_steps = 1000  # Number of simulation steps

simulation = GasSimulation(L, N, m, r, V0, dt)
simulation.simulate(num_steps)
