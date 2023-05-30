import random
import math
from PIL import Image, ImageDraw
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np

class Particle:
    def __init__(self, x, y, vx, vy, r, m):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.m = m


class GasSimulation:
    def __init__(self, L, N, m, r, V0, delta_t):
        self.L = L
        self.N = N
        self.m = m
        self.r = r
        self.V0 = V0
        self.delta_t = delta_t
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
            particle = Particle(x, y, vx, vy, self.r, self.m)

            # Check for intersections with existing particles
            is_valid = all(math.sqrt((p.x - particle.x)**2 + (p.y - particle.y)**2) >= 2 * self.r for p in self.particles)

            if is_valid:
                self.particles.append(particle)

    def movement(self):
        for particle in self.particles:
            particle.x += particle.vx * self.delta_t
            particle.y += particle.vy * self.delta_t
            
    def calculate_pressure(self):
        A = self.L**2
        pressure = self.momentum/(A*self.wall_collisions)
        return pressure
    
    def calculate_temperature(self):
        E = 0
        for i in range (self.N):
            particle = self.particles[i]
            E += 0.5*self.m*np.linalg.norm((particle.vx,particle.vy))**2
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
                dx = particle2.x - particle1.x
                dy = particle2.y - particle1.y
                distance = math.sqrt(dx**2 + dy**2)
                if distance <= 2 * self.r:  # Collision between particles
                    vx1, vy1 = particle1.vx, particle1.vy
                    vx2, vy2 = particle2.vx, particle2.vy
                    particle1.vx = vx1 - (2 * particle2.m * (vx1 - vx2)) / (particle1.m + particle2.m)
                    particle1.vy = vy1 - (2 * particle2.m * (vy1 - vy2)) / (particle1.m + particle2.m)
                    particle2.vx = vx2 - (2 * particle1.m * (vx2 - vx1)) / (particle1.m + particle2.m)
                    particle2.vy = vy2 - (2 * particle1.m * (vy2 - vy1)) / (particle1.m + particle2.m)

            # Check collisions with walls
            particle = self.particles[i]
            if particle.x - particle.r <= 0 or particle.x + particle.r >= self.L:
                particle.vx = -particle.vx
                self.wall_collisions += 1
                self.momentum += 2*particle.m*abs(particle.vx)
            if particle.y - particle.r <= 0 or particle.y + particle.r >= self.L:
                particle.vy = -particle.vy
                self.wall_collisions += 1
                self.momentum += 2*particle.m*abs(particle.vy)
    
    
        #     #Get energy
        #     self.particles.E += 0.5*self.m*np.linag.norm(self.particle.vx,self.particle.vy)**2
            
        # #Get temperature
        # self.particles.T = (2/3)*self.particles.E/(self.N *1.380649*10**-23) -273.15
                


    def simulate(self, num_steps):
        self.initialize_particles()
        

        # frames = []

        for _ in range(num_steps):
            # Create a new frame for each step
            # frame = self.create_frame_image()
            # frames.append(frame)

            self.movement()
            self.collisions()
            

        # self.save_animation(frames)
        
        #Obtain final velocities
        velocities = [np.linalg.norm((p.vx,p.vy)) for p in self.particles]
        pressure = self.calculate_pressure()
        temperature = self.calculate_temperature()
        momentum = self.momentum
        ideal_gas = self.ideal_gas(temperature,pressure)
        
        print(f"Momentum:{momentum}")
        print(f"Pressure:{pressure}")
        print(f"Temperature:{temperature}")
        print(f"Ideal gas:{ideal_gas}")
        
        
        sns.kdeplot(velocities)
        
        plt.xlabel('Velocity')
        plt.ylabel('Density')
        plt.title('Velocity Distribution')
        plt.show()
    

    def create_frame_image(self):
        scale_factor = 40  

        image_width = int(self.L * scale_factor)
        image_height = int(self.L * scale_factor)

        image = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(image)

        for particle in self.particles:
            x1 = int((particle.x - particle.r) * scale_factor)
            y1 = int((particle.y - particle.r) * scale_factor)
            x2 = int((particle.x + particle.r) * scale_factor)
            y2 = int((particle.y + particle.r) * scale_factor)

            draw.ellipse([(x1, y1), (x2, y2)], fill='blue')

        return image

    def save_animation(self, frames):
        gif_path = 'gas_simulation.gif'
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        print(f"GIF animation saved as '{gif_path}'")
        

L = 4  # Box volume (L^2)
N = 100  # Number of particles
m = 5  # Particle mass
r = 0.07  # Particle radius
V0 = 10  # Initial velocity
delta_t = 0.005  # Time step
num_steps = 100  # Number of simulation steps

simulation = GasSimulation(L, N, m, r, V0, delta_t)
simulation.simulate(num_steps)
