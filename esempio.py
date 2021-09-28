from __future__ import print_function
import openmm as mm
import openmm.app as app
import openmm.unit as u
#from reducedstatedatareporter import ReducedStateDataReporter
#import simulation as sim
import numpy as np
import os, sys
import parmed as pmd
import json
from sys import platform
import unittest
import csv


numParticles = 100

### setting LJ parameters (they can be one for each couple)
epsilon_r = np.full(numParticles, 1., dtype="float64")
sigmas_r=np.full(numParticles, 1., dtype="float64")
sigmaAR_r = np.zeros((numParticles, numParticles), dtype="float64")
epsilonAR_r = np.zeros((numParticles, numParticles), dtype="float64")

for i in range(numParticles):
    for j in range(i,numParticles):
        sigmaAR_r[i][j] = (sigmas_r[i]+sigmas_r[j])/2.0
        sigmaAR_r[j][i] = sigmaAR_r[i][j]

for i in range(numParticles):
    for j in range(i,numParticles):
        epsilonAR_r[i][j] = (epsilon_r[i]+epsilon_r[j])/2.0
        epsilonAR_r[j][i] = epsilonAR_r[i][j]

epsilonLST_r = (epsilonAR_r).ravel().tolist()
sigmaLST_r   = (sigmaAR_r).ravel().tolist()
masses_r= np.full(numParticles, 1., dtype="float64")

########### Building system + particles
system = mm.System()
positions = np.empty((numParticles, 3)) # matrix 3*numParticles (3D)

cutoff_r=10.*max(sigmaLST_r) #cutoff distance for LJ
########### Building forces
box_edge_r=1000.
system.setDefaultPeriodicBoxVectors(mm.Vec3(box_edge_r, 0, 0), mm.Vec3(0, box_edge_r, 0), 
    mm.Vec3(0, 0, box_edge_r))

#harmonic   
k=0.0
#trying to divide in regions
k_arr= np.full(numParticles, 10., dtype="float64")
k_arr[50]=1.
el_force = mm.HarmonicBondForce()
for i in range(numParticles-1):
    el_force.addBond( i, i+1, 0.5, k_arr[i]) #particle 1, particle 2, length at rest, k elastic (unit: kJ/mol/nm^2)

#lennard-jones
lj_force = mm.CustomNonbondedForce('4*eps*((sig/r)^12-(sig/r)^6); eps=epsilon(type1, type2); sig=sigma(type1, type2)')
lj_force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
lj_force.setCutoffDistance(min(box_edge_r*0.49*u.nanometers, cutoff_r*u.nanometers))
lj_force.addTabulatedFunction('epsilon', mm.Discrete2DFunction(numParticles, numParticles, 
    epsilonLST_r))
lj_force.addTabulatedFunction('sigma', mm.Discrete2DFunction(numParticles, numParticles,
    sigmaLST_r))
lj_force.addPerParticleParameter('type')


# set the initial particle parameters
for i in range(numParticles):
    system.addParticle(masses_r[i]*u.amu)
    positions[i] = [i, 0.1*i, -0.3*i] + 10*np.random.rand(3)
    lj_force.addParticle([i])
    
system.addForce(lj_force)
system.addForce(el_force)
tol=0.3
maxIter=0.

#integ = mm.LangevinIntegrator(300.0, 1.0, 0.1)
integ = mm.VerletIntegrator(0.001)
#integ = mm.VariableVerletIntegrator(0.1)


context = mm.Context(system, integ, mm.Platform.getPlatformByName('CPU'))
context.setPositions(positions)
state = context.getState(getEnergy=True, getForces=True, getPositions=True)
#print('positions before minimization: ', np.array(state.getPositions()/u.nanometer))
mm.LocalEnergyMinimizer.minimize(context, tol, maxIter)
context.setVelocitiesToTemperature(0) #added

state = context.getState(getEnergy=True, getForces=True, getPositions=True)
#print('positions after minimization: ',np.array(state.getPositions()/u.nanometer))
print('potential energy after minimization: ', state.getPotentialEnergy())
print('kinetic energy after minimization: ', state.getKineticEnergy())
print('total energy after minimization: ', state.getKineticEnergy() + state.getPotentialEnergy())

#integ.step(10)
#state = context.getState(getEnergy=True, getForces=True, getPositions=True)
#print('positions after integration: ',np.array(state.getPositions()/u.nanometer))

### write on xyz file
with open('sim.xyz', 'a', newline='') as file:
    for j in range(1000): #number of blocks of integration
        file.write(str(numParticles))
        file.write('\n')
        file.write('\n')
        integ.step(100) #steps for each block
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        #print('============================================')
        #print('blocco: ',j)
        #print('potential energy after integration: ', state.getPotentialEnergy())
        #print('kinetic energy after integration: ', state.getKineticEnergy())
        #print('total energy after integration: ', state.getKineticEnergy() + state.getPotentialEnergy())
        #ora divido per nanometri perchè altrimenti compare unità di misura in file xyz
        alpha=state.getPositions(asNumpy=True)/u.angstrom
        for i in range(numParticles):
            file.write(f"C {alpha[i][0]} {alpha[i][1]} {alpha[i][2]}\n")


state = context.getState(getEnergy=True, getForces=True, getPositions=True)

print('============================================')
print('potential energy after integration: ', state.getPotentialEnergy())
print('kinetic energy after integration: ', state.getKineticEnergy())
print('total energy after integration: ', state.getKineticEnergy() + state.getPotentialEnergy())



### write on csv file
'''
with open('sim.csv', 'a', newline='') as file:
    for j in range(1000):
        #writer = csv.writer(file, delimiter=',')
        integ.step(1)
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        #ora divido per nanometri perchè altrimenti compare unità di misura in file xyz
        alpha=np.array(state.getPositions()/u.angstrom)
        for i in range(numParticles):
            file.write(f"{alpha[i][0]}, {alpha[i][1]}, {alpha[i][2]}\n")
'''