import numpy as np
import random
import math
import imageio
import matplotlib.pyplot as plt


class Particles():
    def __init__(self,x,y,vx,vy,r,m):
        self.x = x
        self.y = y
        self.vx = vx 
        self.vy = vy 
        self.r = r
        self.m = m
        
class Gas_simulation():
    def __init__(self,L,N,m,r,V0,dt):
        self.L = L
        self.m = m
        self.N = N
        self.r = r 
        self.V0 = V0
        self.dt = dt
        self.particles = []
    
       
        
    def initializate_particles(self):
        while len(self.particles)<self.N:
            x = random.uniform(self.r,self.L-self.r)
            y = random.uniform(self.r,self.L-self.r)
            theta = random.uniform(0,2*np.pi)
            vx = self.V0*math.cos(theta)
            vy = self.V0*math.sin(theta)
            particle = Particles(x, y, vx, vy, self.r, self.m)
            
            valid = all(math.sqrt((p.x - particle.x)**2 + (p.y - particle.y)**2) >= 2 * self.r for p in self.particles)
            
            if valid:
                self.particles.append(particle)
                
    def movement(self):
        for particle in self.particles:
            particle.x += particle.vx*self.dt 
            particle.y += particle.vy*self.dt
            
    def collisions(self):
        for i in range(self.N):
            for j in range(i+1,self.N):
                p1 = self.particles[i]
                p2 = self.particles[j]
                dis = math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2)
                
                
                #Collision between two particles
                if dis <= 2*self.r:
                    vx1, vy1 = p1.vx, p1.vy
                    vx2, vy2 = p2.vy, p2.vy
                    p1.vx = vx1 - (2*p2.m *(vx1-vx2))/(p1.m + p2.m)
                    p1.vy = vy1 - (2*p2.m*(vy1-vy2))/(p1.m + p2.m)
                    p2.vx = vx2 - (2*p1.m*(vx2-vx1))/(p1.m + p2.m)
                    p2.vy = vy2 - (2*p1.m*(vy2-vy1))/(p1.m+p2.m)
                    
                    
            #Check collision with the walls
            particle = self.particles[i]
            if particle.x <= particle.r or particle.x >= self.L+particle.r:
                particle.vx = -particle.vx
            
            if particle.y <= particle.r or particle.y >= self.L+particle.r:
                particle.vy = -particle.vy
                
    def simulation(self,it):
        self.initializate_particles()
        frames = []
        for _ in range(it):
            self.movement()
            self.collisions
            
            frame = self.create_frame_image()
            frames.append(frame)
            
            
        self.save_animation(frames)
        
        
    def create_frame_image(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal', adjustable='box')
        for particle in self.particles:
            circle = plt.Circle((particle.x, particle.y), particle.r, color='b')
            ax.add_artist(circle)
        plt.axis('off')

        # Save the figure as an image and return it
        image_path = 'frame.png'
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return image_path
    
    
    def save_animation(self, frames):
        gif_path = 'gas_simulation.gif'
        with imageio.get_writer(gif_path, mode='I', duration=20) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)

        print(f"GIF animation saved as '{gif_path}'")    
    
    
L = 5  # Box volume (L^2)
N = 100  # Number of particles
m = 1  # Particle mass
r = 0.1  # Particle radius
V0 = 1  # Initial velocity
delta_t = 0.01  # Time step
num_steps = 1000  # Number of simulation steps

simulation = Gas_simulation(L, N, m, r, V0, delta_t)
simulation.simulation(num_steps)
                    
                    
                
        
        
        
        
        
        
        
        