#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:40:00 2022

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.special as sps
from scipy import integrate
import pdb

#%% - 
theta=np.linspace(0,2*np.pi,1000) #Discretization of a circumference
R=3 #Radius 

p1=np.array([0,R]) #Position of the center

p2=np.array([np.sin(theta),np.cos(theta)])*R #array of the points of circumference

plt.plot(p2[0,:], p2[1,:]); plt.show()
plt.show()
dist=p2.copy()
dist[1,:]-=R

C=np.linalg.norm(dist,axis=0)
plt.plot(C)
plt.title("distance to one of the points in the circumference")


C=np.sqrt(2*R**2*(1-np.cos(theta))); plt.plot(C)


np.log((np.cos(np.pi/8)+np.sin(np.pi/8))/(np.cos(np.pi/8)-np.sin(np.pi/8)))



#%% - PV in 2D (Principal value integral, basically we exclude the point with singularity)
#This section is useful to prove the principal value integral with an easy 2D example
c=0
integral=0
for i in p2.T[1:-1,:]: #We exclude the singularity 
    x_c=i[0]
    y_c=i[1]
    r=-np.array([x_c,y_c])+np.array([0,R])
    r=-r
    d=np.linalg.norm(r)
    er=r/d
    
    grad_G=1/d
    n=np.array([x_c, y_c])/R
    integral+=np.dot(er,n)*grad_G
    c+=1

integral=integral*R/(len(theta-2))
print(integral)

#%%


def get_ref_integral(inc_s,R):
    G = lambda theta :(inc_s**2 + (2*R*np.sin(theta))**2)**-0.5 
    return(integrate.quad(G,0,2*np.pi)[0]*R/(4*np.pi))

def get_ellip(inc_s,R):
    k=(inc_s**2/(4*R**2)+1)**-0.5
    
    return(sps.ellipk(k**2)*k/(2*np.pi))

def get_ref_integral2(inc_s,R):
    G = lambda theta :(inc_s**2 + (2*R*np.sin(theta))**2)**-0.5 
    return(integrate.quad(G,0,np.pi/2)[0]*R/np.pi)

def get_ref_integral3(inc_s,R):
    G = lambda theta :(inc_s**2 + 4*R**2-(2*R*np.sin(theta))**2)**-0.5 
    return(integrate.quad(G,0,np.pi/2)[0]*R/np.pi)

def get_ellip_man(inc_s,R,phi):
    k2=(1+inc_s**2/(4*R**2))**-0.5
    k=(inc_s**2/(4*R**2)+1)**-0.5
    G = lambda theta :(inc_s**2 + 4*R**2*np.sin(theta)**2)**-0.5
    return(integrate.quad(G,phi/2,np.pi*2-phi/2)[0]*R/(4*np.pi))

def get_point_point(inc_s, R):
    return((4*np.pi*(inc_s**2+R**2)**0.5)**-1*2*np.pi*R)

class cyl_full_integral():
    def __init__(self,axial_disc, L, R):
        
        self.Ns=axial_disc
        self.h=L/self.Ns
        self.s=np.linspace(self.h/2,L-self.h/2,self.Ns)
        self.L=L
        self.R=R
        self.k=np.zeros(self.Ns)
        #arrays to return

    
    def ellip_integral_SL_singular(self):
        phi=np.arcsin(self.h/(2*self.R))
        self.phi=phi
        return(0.44*self.h+get_ellip_man(self.h,self.R, phi))
    
    def ellip_integral_SL_nonsingular(self, inc_s):
        k=(inc_s**2/(4*self.R**2)+1)**-0.5
        return(sps.ellipk(k**2)*k/(2*np.pi))

    def get_point_wise_approximation(self,inc_s, R):
        #returns the value of G_ax
        #pdb.set_trace()
        if inc_s<=self.h/2:
            return(self.ellip_integral_SL_singular())
        else:
            return(self.ellip_integral_SL_nonsingular(inc_s))
    
    def get_array_point_wise(self):
        F=np.zeros(self.Ns)
        R=self.R
        L=self.L
        for i in range(self.Ns):
            inc_s=self.s[i]-L/2
            F[i]=self.get_point_wise_approximation(inc_s, R)*self.h
        self.F=F
        
    #TO COMPUTE NON INTEGRATED ARRAYS
    
    def not_integrated(self,s_0):
        point_wise=np.zeros(self.Ns)
        ellip_wise=np.zeros(self.Ns)
        Gjerde=np.zeros(self.Ns)
        for i in range(self.Ns):
            inc_s=np.linalg.norm(self.s[i]-s_0)
            #pdb.set_trace()
            ellip_wise[i]=self.get_point_wise_approximation(inc_s, self.R)
            point_wise[i]=get_point_point(inc_s, self.R)
            init=np.array([self.s[i]-self.h/2,0])
            end=np.array([self.s[i]+self.h/2,0])
            Gjerde[i]=sing_term2(init, end, np.array([s_0, self.R]), R)*2*np.pi*R
            
        self.ellip_point=ellip_wise
        self.point_point=point_wise
        self.Gjerde=Gjerde
        return()
    
    #INTEGRATION FUNCTIONS
    def integrate_0D(self):
        #Integrates the 0D approximation at the vessel wall
        F_point=np.zeros(self.Ns)
        R=self.R
        L=self.L
        for i in range(self.Ns):
            si=self.s[i]
            for j in range(self.Ns):
                sj=self.s[j]
                inc_s=np.abs(sj-si)
                F_point[i]+=get_point_point(inc_s,R)*self.h
        self.SL_0D=F_point
    
    def integrate_2D(self):
        #Integrates the 2D approximation at the vessel wall
        self.SL_2D=np.zeros(self.Ns)
        #pdb.set_trace()
        for i in range(self.Ns):
            si=self.s[i]
            for j in range(self.Ns):
                sj=self.s[j]
                self.SL_2D[i]+=self.get_point_wise_approximation(np.abs(si-sj), R)*self.h
                
    def integrate_1D(self):
        #Returns the analytically integrated value of the 1D approximation
        L=self.L
        R=self.R
        self.SL_1D=np.zeros(self.Ns)
        
        for i in range(self.Ns): 
            #pdb.set_trace()
            si=self.s[i]
            init=np.array([0,0])
            end=np.array([0,L])
            x=np.array([R,si])
            #pdb.set_trace()
            pp=sing_term2(init, end, x, R)*2*np.pi*R
            self.SL_1D[i]=pp
                    
def sing_term2(init, end, x, Rv):
    tau=(end-init)/np.linalg.norm(end-init)
    L=np.linalg.norm(end-init)
    a=x-init
    b=x-end
    s=np.sum(a*tau)
    d=np.linalg.norm(a-tau*np.sum(a*tau))
    
    
    rb=np.linalg.norm(b)
    ra=np.linalg.norm(a)
    G=np.log((rb+L-np.dot(a,tau))/(ra-np.dot(a,tau)))/(4*np.pi)
    return(G)
                    
class line_full_integral():
    def __init__(self,axial_disc, L, R):
        self.Ns=axial_disc
        self.h=L/self.Ns
        self.s=np.linspace(self.h/2,L-self.h/2,self.Ns)
        self.L=L
        self.R=R
        #arrays to return
        self.SL=np.zeros(self.Ns, dtype=float)
        self.DL=np.zeros(self.Ns)
        
def approx_dipoles(L,R, disc):
    s=np.linspace(0,L,disc)
    ls=L-s
    dip=R**2/4*(-s/(R**2+s**2)**1.5 + ls/(R**2+ls**2)**1.5)
    return(dip)

def dip2(L,R):
    s=np.linspace(0,L,1000)
    ls=L-s
    dip=R**3/2*(-s/(R**2+s**2)**1.5 + ls/(R**2+ls**2)**1.5)
    return(dip)

#%%
R=0.5
L=10

 #%% - Something dipoles
plt.plot(dip2(1,0.1))
plt.title("dipoles with constant gradient")

#%%
def dsc(i):
    return(7*i**3)
sols=dict()

for i in np.arange(5):
    disc=dsc(i+1)
    t=cyl_full_integral(disc,L,R)
    t.integrate_2D()
    sols["{}".format(disc)]=t.SL_2D
    

#%%
plt.figure()
for i in np.arange(3):
    disc=dsc(i+2)
    
    plt.plot(np.linspace(0,L,disc), sols["{}".format(disc)], label="{}".format(disc))

u=cyl_full_integral(100,L,R)
u.integrate_1D()
plt.plot(np.linspace(0,L,100), u.SL_1D, linewidth=3,label='Int line')
#plt.plot(np.linspace(0,L,disc), sols["{}".format(disc)]+approx_dipoles(L,R, disc), linewidth=3,label="Full Green's")
plt.plot(np.linspace(0,L,disc), sols["{}".format(disc)], linewidth=3,label="Ref")
plt.legend()
plt.show()

#%%
Full=sols["{}".format(disc)]+approx_dipoles(L,R, disc)

u=cyl_full_integral(disc,L,R)
u.integrate_1D()
plt.plot((Full-u.SL_1D)/Full)

print(np.sum((Full-u.SL_1D)/Full)/875)
    
#%%
point=0.3*L
ref=cyl_full_integral(1000,L,R)
ref.not_integrated(point)
ref_point=ref.ellip_point

ref.integrate_2D()
ref_SL=ref.SL_2D
#%%
N=10
#%%
t=cyl_full_integral(N,L,R)

  
u=line_full_integral(N,L,R)

t.not_integrated(point)
plt.plot(t.s-point,t.ellip_point, label="2D"); 
plt.plot(t.s-point,t.point_point, label="0D"); 
plt.plot(t.s-point,t.Gjerde/t.h, label="1D");
plt.plot(ref.s-point,ref_point, label="ref")
plt.xlabel("$\Delta$s")
plt.title("Point wise SL potential ingluence SL($\Delta$s) L={} \n aspect ratio={}".format(L, L/(2*R)))
plt.legend()
#%%

t.integrate_0D()
t.integrate_1D()
t.integrate_2D()
#%%
#Even with the 2 dipole is added there is a 20% difference???
#plt.scatter(t.s,t.SL_2D,label="2D integration")
plt.plot(t.s,t.SL_1D,'-.' ,label="1D integration")
plt.plot(t.s,t.SL_0D, label='0D integration')
plt.plot(ref.s,ref_SL, label="ref")
plt.title("integrated SL for aspect ratio of {}".format(L/(2*R)))
plt.legend()
#%%        

err=np.zeros(0)
for i in np.linspace(0.01,1,100):
    inc_s=i
    R=1
    print(get_ref_integral(inc_s,R))
    print(get_ref_integral2(inc_s,R))
    print(get_ref_integral3(inc_s,R))
    print(get_ellip(inc_s,R))
    print(get_ellip_man(inc_s,R),"\n")
    k=(inc_s**2/(4*R**2)+1)**-0.5
    err=np.append(err, (get_ellip(inc_s,R)-get_ellip_man(inc_s,R))/get_ellip(inc_s,R))
    
def get_k(inc_s, R):
    return((inc_s**2/(4*R**2)+1)**-0.5)
    
#%%
    
def integral_along_cylinder(L,pos_or, R, disc_axial):
    s=np.linspace(-L+pos_or*L+L/(2*disc_axial),L-pos_or*L-L/(2*disc_axial), disc_axial)
    pos=np.argmin((s-L*pos_or)**2)
    s=np.delete(s,pos)
    c=0
    inc_s=L/disc_axial
    #pdb.set_trace()
    F=np.zeros(len(theta))
    for i in s:
        k=(i**2/(4*R**2)+1)**-0.5
        F[c]=k*sps.ellipk(k**2)/(2*np.pi)
        c+=1
    return(F)
    

#def singular_single_layer(L,disc_axial,R):
    

a=integral_along_cylinder(10,0.25,1,1000)
plt.plot(a)
plt.title("F(s)")