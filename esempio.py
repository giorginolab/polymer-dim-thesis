from __future__ import print_function
import openmm as mm
import openmm.app as app
import openmm.unit as u
#from reducedstatedatareporter import ReducedStateDataReporter
import numpy as np
import os, sys
import parmed as pmd
import json
from sys import platform
import unittest
import csv
import sys
import yaml

filename = str(sys.argv[1])

with open(filename) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

N=int(params["num_particles"])
T=float(params["temperature"])
k=float(params["k_contig"]) # k elastico contigui
k_h=float(params["k_helix"]) # k elastico elica
l_0=float(params["length_rest"]) 
l_0_h=float(params["length_rest_helix"])
m=float(params["mass"])
eps=float(params["epsilon"])
sigma=float(params["sigma"])
n_blocks=int(params["n_blocks"])
n_steps=int(params["n_steps"])
timestep=float(params["timestep"])

print("Numero di monomeri: ", N)
### setting LJ parameters (they can be one for each couple)
epsilon_r = np.full(N, eps, dtype="float64")
sigmas_r=np.full(N, sigma, dtype="float64")
sigmaAR_r = np.zeros((N, N), dtype="float64")
epsilonAR_r = np.zeros((N, N), dtype="float64")

for i in range(N):
    for j in range(i,N):
        sigmaAR_r[i][j] = (sigmas_r[i]+sigmas_r[j])/2.0
        sigmaAR_r[j][i] = sigmaAR_r[i][j]

for i in range(N):
    for j in range(i,N):
        epsilonAR_r[i][j] = (epsilon_r[i]+epsilon_r[j])/2.0
        epsilonAR_r[j][i] = epsilonAR_r[i][j]

epsilonLST_r = (epsilonAR_r).ravel().tolist()
sigmaLST_r   = (sigmaAR_r).ravel().tolist()
masses_r= np.full(N, m, dtype="float64")

########### Building system + particles
system = mm.System()
positions = np.empty((N, 3)) # matrix 3*N (3D)

cutoff_r=10.*max(sigmaLST_r) #cutoff distance for LJ
########### Building forces
box_edge_r=1000.
system.setDefaultPeriodicBoxVectors(mm.Vec3(box_edge_r, 0, 0), mm.Vec3(0, box_edge_r, 0), 
    mm.Vec3(0, 0, box_edge_r))

#harmonic   

k_arr= np.full(N, k, dtype="float64")
k_arr_d= np.full(N-3, k_h , dtype="float64") #per alpha elica
el_force = mm.HarmonicBondForce()
for i in range(N-1):
    el_force.addBond( i, i+1, 0.38, k_arr[i]) #particle 1, particle 2, length at rest, k elastic (unit: kJ/mol/nm^2)

#elastic force between i and i+3
#for i in range(N-4):
#    el_force.addBond( i, i+3, 0.516, k_arr_d[i]) #particle 1, particle 2, length at rest, k elastic (unit: kJ/mol/nm^2)


#lennard-jones
lj_force = mm.CustomNonbondedForce('4*eps*((sig/r)^12-(sig/r)^6); eps=epsilon(type1, type2); sig=sigma(type1, type2)')
lj_force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
lj_force.setCutoffDistance(min(box_edge_r*0.49*u.nanometers, cutoff_r*u.nanometers))
lj_force.addTabulatedFunction('epsilon', mm.Discrete2DFunction(N, N, 
    epsilonLST_r))
lj_force.addTabulatedFunction('sigma', mm.Discrete2DFunction(N, N,
    sigmaLST_r))
lj_force.addPerParticleParameter('type')


# set the initial particle parameters
for i in range(N):
    system.addParticle(masses_r[i]*u.amu)
    positions[i] = [i, 0.1*i, -0.3*i] + 10*np.random.rand(3)
    lj_force.addParticle([i])
    
system.addForce(lj_force)
system.addForce(el_force)
tol=0.3
maxIter=0.

integ = mm.LangevinIntegrator(T, 1.0, timestep)
#integ = mm.VerletIntegrator(0.001)
#integ = mm.VariableVerletIntegrator(0.1)

context = mm.Context(system, integ, mm.Platform.getPlatformByName('CPU'))
context.setPositions(positions)
state = context.getState(getEnergy=True, getForces=True, getPositions=True)
#print('positions before minimization: ', np.array(state.getPositions()/u.nanometer))
mm.LocalEnergyMinimizer.minimize(context, tol, maxIter)
context.setVelocitiesToTemperature(0) 

state = context.getState(getEnergy=True, getForces=True, getPositions=True)
print('potential energy after minimization: ', state.getPotentialEnergy())
print('kinetic energy after minimization: ', state.getKineticEnergy())
print('total energy after minimization: ', state.getKineticEnergy() + state.getPotentialEnergy())

### write on xyz and csv file
# name of the file tells number of particles, k contiguous and temperature
name = 'num' + str(N) + 'k' + str(k) + 'temp' + str(T)
with open(name + '.xyz', 'w', newline='') as file, open(name + '.csv', 'w', newline='') as file_csv:
    for j in range(n_blocks): #number of blocks of integration
        file.write(str(N))
        file.write('\n')
        file.write('\n')
        integ.step(n_steps) #steps for each block
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        alpha=state.getPositions(asNumpy=True)/u.angstrom
        for i in range(N):
            file.write(f"C {alpha[i][0]} {alpha[i][1]} {alpha[i][2]}\n")
            file_csv.write(f"{alpha[i][0]} {alpha[i][1]} {alpha[i][2]} ")
        file_csv.write("\n")


state = context.getState(getEnergy=True, getForces=True, getPositions=True)

print('============================================')
print('potential energy after integration: ', state.getPotentialEnergy())
print('kinetic energy after integration: ', state.getKineticEnergy())
print('total energy after integration: ', state.getKineticEnergy() + state.getPotentialEnergy())

print("simulation: ", n_blocks, " frames, ", n_steps , " timesteps each")
