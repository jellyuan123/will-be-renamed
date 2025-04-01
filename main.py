import numpy as np
from setup import *
from accelBins import *

print("This classifier has read in data from 10 unknown tracks, and will now classify these tracks.")

classes = ['bird', 'airplane']

# reading in data and initializing
unclassifieds, bird_likelihoods, plane_likelihoods, bird_accels, plane_accels = readAndSortData()
speed = np.linspace(0, 200, 400)

# Defining transition probabilities
P_birdbird = 0.9
P_birdplane = 0.1
P_planebird = 0.1
P_planeplane = 0.9

P_bird_given_o_prev = 0.5
P_plane_given_o_prev = 0.5

i = 1
for current_unclassified in unclassifieds: # this loop represents the recursive bayesian estimation function 

        # this is my implementation of the recursive bayesian function
        for x in range(0, 599):

                observation = current_unclassified[x]
                accel = current_unclassified[x] - current_unclassified[x-1]

                # was having a lot of issues with NaNs
                if not(np.isnan(observation) or np.isnan(accel)):

                        bird_vlikelihood = bird_likelihoods[np.abs(speed - observation).argmin()]
                        plane_vlikelihood = plane_likelihoods[np.abs(speed - observation).argmin()]

                        bird_alikelihood, plane_alikelihood = get_accel_likelihoods(bird_accels, plane_accels, accel)

                        combined_bird_likelihood = bird_vlikelihood * bird_alikelihood
                        combined_plane_likelihood = plane_vlikelihood * plane_alikelihood

                        P_bird = P_bird_given_o_prev * P_birdbird + P_plane_given_o_prev * P_planebird
                        P_plane = P_bird_given_o_prev * P_birdplane + P_plane_given_o_prev * P_planeplane

                        P_accel_velocity = combined_bird_likelihood * P_bird + combined_plane_likelihood * P_plane

                        if P_accel_velocity > 0:
                                P_bird_given_accel_velocity = (combined_bird_likelihood * P_bird) / P_accel_velocity
                                P_plane_given_accel_velocity = (combined_plane_likelihood * P_plane) / P_accel_velocity
                        else:
                                P_bird_given_accel_velocity = 1e-10
                                P_plane_given_accel_velocity = 1e-10

                        P_bird_given_o_prev = P_bird_given_accel_velocity
                        P_plane_given_o_prev = P_plane_given_accel_velocity

                        

        if P_bird_given_accel_velocity > P_plane_given_accel_velocity:
                final_classification = 'bird'
        else:
                final_classification = 'airplane'

        print(f"Classification for track {i}: {final_classification}")
        i += 1
        
        