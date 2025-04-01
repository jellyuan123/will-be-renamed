import numpy as np
from accelBins import *

def readAndSortData():
        
        with open('likelihood.txt', 'r') as file:
                all_likelihoods = []
                for line in file:
                        file_row = [float(num) if num != 'NaN' else np.nan for num in line.split()]   # Convert split elements to integers
                        all_likelihoods.append(file_row) 

        bird_likelihoods = all_likelihoods[0]  
        plane_likelihoods = all_likelihoods[1]  

        with open('dataset.txt', 'r') as file:
                all_velocities = []
                for line in file:
                        file_row = [float(num) if num != 'NaN' else np.nan for num in line.split()]  
                        all_velocities.append(file_row) 

        bird_velocities = all_velocities[:10] 
        plane_velocities = all_velocities[10:]  

        with open('testing.txt', 'r') as file:
                unclassifieds = []
                for line in file:
                        file_row = [float(num) if num != 'NaN' else np.nan for num in line.split()] 
                        unclassifieds.append(file_row)  

        bird_accels = accelBins()
        plane_accels = accelBins()

        bird_accels.initialize(get_accels(bird_velocities))
        plane_accels.initialize(get_accels(plane_velocities))


        return unclassifieds, bird_likelihoods, plane_likelihoods, bird_accels, plane_accels


# computes all the acceleratiosn
def get_accels(velocities):
        all_accel = []

        for row in velocities:
                row = [value for value in row if not np.isnan(value)]

                if len(row) > 1:
                        accelerations = np.diff(row)
                        all_accel.extend(accelerations)  

        return all_accel


# returns the likelihood for a given acceleration for both planes and birds
def get_accel_likelihoods(bird_accels, plane_accels, accel_given):

        P_accel_given_bird = bird_accels.getFreq(accel_given)     
        P_accel_given_plane = plane_accels.getFreq(accel_given) 

        P_bird = bird_accels.getLen() / (bird_accels.getLen() + plane_accels.getLen())
        P_plane = plane_accels.getLen() / (bird_accels.getLen() + plane_accels.getLen())

        P_accel = P_accel_given_bird * P_bird + P_accel_given_plane * P_plane

        if P_accel > 0:
                # using Bayes Theorem
                P_bird_given_accel = (P_accel_given_bird * P_bird) / P_accel
                P_plane_given_accel = (P_accel_given_plane * P_plane) / P_accel

                return P_bird_given_accel, P_plane_given_accel
        
        else:
                return 0, 0