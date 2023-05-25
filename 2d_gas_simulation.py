import numpy as np
import random
import math


class Particles():
    def __init__(self,x,y,vx,vy,r,m):
        self.x = x
        self.y = y
        self.vx = vx 
        self.vy = vy 
        self.r = r
        self.m = m
        
class Gas_simulation():
    def __init__(self,L,m,N,r,V0,dt):
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
            
    def collitions(self):
        for i in range(self.N):
            for j in range(i+1,self.N):
                p1 = self.particles[i]
                p2 = self.particles[j]
                dis = math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2)
                
                if dis <= 2*self.r:
                    vx1, vy1 = p1.vx, p1.vy
                    vx2, vy2 = p2.vy, p2.vy
                    
                
        
        
        
        
        
        
        
        