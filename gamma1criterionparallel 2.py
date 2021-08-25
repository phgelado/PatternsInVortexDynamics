import time
import concurrent.futures
import pandas as pd 
from sklearn.cluster import MeanShift
from IPython.display import clear_output
import numpy as np
from numpy.linalg import norm
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
from IPython import display


#importing data 
#Import Velocity File Including Computed Q-Criterion from separate files for each time step.

t0 = time.time()

data = []

velocityAll = pd.DataFrame(data, columns = {'Frame','X', 'Y', 'U', 'V', 'Q-criterion'})

print("Importing velocity files...")

initialFrame = 0

finalFrame = 400

frameStep = 1

def import_frames(frameNumber):

    global velocityAll

    #input file paths
    input_file = '~/Documents/Python Scripts/Velocity/velocity0.'
    input_file += str(int(frameNumber))
    input_file += '.csv'

    df = pd.read_csv(input_file)
    
    df = df.drop(["U:2","Points:2"], axis = 1)
    
    df = df.rename(columns={"Points:0": "X", "Points:1": "Y","U:0": "U", "U:1": "V"})
    
    df['Frame'] = frameNumber
    
    velocityAll = velocityAll.append(df)

    print('Frame Status Update --> ', round(100*(frameNumber-initialFrame)/(finalFrame-initialFrame),2),'%')
    display.clear_output(wait=True)


print("Velocity files imported.")

#Gamma-1 Function Code to locate Vortex Cores

# gamma-1 function definition

def gamma1(point,veldata,n_neighbs):
       
    neighbours = closestneighbours(point,veldata,n_neighbs)
    sigma = 0
    N = len(neighbours)
       
    convolution1 = 0
    convolution2 = 0
    gamma1 = 0
       
    for i in range(N):
           
        PMx = neighbours[i][0]
        PMy = neighbours[i][1]
        UMx = neighbours[i][2]
        UMy = neighbours[i][3]
           
        divisor = (norm([PMx,PMy])*norm([UMx,UMy]))
           
        if divisor!=0:
            convolution1 += (PMx * UMy)/divisor
            convolution2 += (PMy * UMx)/divisor
       
    gamma1 += (convolution1-convolution2)/N
       
    return gamma1    


# return de x closest neighbours to a point in a velocity data array
def closestneighbours(point,veldata,n_neighbs):
    neighbours = list()
    for i in range(len(veldata)):
        vector = np.array([veldata[i][0] - point[0], veldata[i][1] - point[1]])
        distance = norm(vector)
        if distance < 0.06: #only added closepoints that are less then a certain distance
            velocity = np.array([veldata[i][2], veldata[i][3]])
            neighbours.append([veldata[i][0],veldata[i][1],veldata[i][2],veldata[i][3]])
    neighbours.sort(key=itemgetter(0))
    return neighbours[1:n_neighbs+1]
                       
def process_frames(j): #for i in range number of frames

    velocity = pd.DataFrame(columns = {'X','Y','U','V','Q-criterion'})

    velocity = velocityAll.loc[velocityAll['Frame']==j]
   
    DATA = velocity[['X','Y','U','V','Q-criterion']].values.tolist()

    #Condition to execute the gamma-1 function just around points with more than the Q-Criterion limit
    qcriterionlimit = 25 #Q-criterion limit to search for cores condition

    Rdata = list() #results

    #Computing the gamma-1 function

    for i in range(len(DATA)):
        if DATA[i][4]>=qcriterionlimit: #Q-criterion limit to search for cores conditional statement
            Rdata.append(DATA[i] + [gamma1(DATA[i], DATA, 50)])
            print(j," frames have been completed out of ", 17, ' frames.')
            print('Frame Status Update --> ', round(100*i/len(DATA),2),'%')
            t1 = time.time()
            total = t1-t0
            print('This program has been running for ',round(total,2), 'seconds')
            display.clear_output(wait=True)
   
    centre = np.array(Rdata)
    centre = pd.DataFrame(data = centre, columns =['X','Y','U','V','Q-criterion','Gamma'])
    clusteringcentres = centre
   
    # #Gamma 1 Core Condition
    # centre['Gamma'] = centre['Gamma'].abs()
    # centre = centre[centre['Gamma'].between(0.85,1)]

    # if centre.empty:
    #     clusteringcentres = centre
    # else:

    #     clustering = centre[['X','Y']].to_numpy()

    #     ms = MeanShift(bandwidth = 0.01)
    #     ms.fit(clustering)
    #     clusteringcentres = ms.cluster_centers_
    #     clusteringcentres = pd.DataFrame(data=clusteringcentres)
    #     clusteringcentres = clusteringcentres.rename(columns ={0:'X',1:'Y'})
    #     clusteringcentres['Frame'] = j

    #output file paths
    output_file = '~/Documents/Python Scripts/Gamma1/gamma1frame_'
    output_file += str(j)
    output_file += '.csv'

    clusteringcentres.to_csv(output_file, index= False)
   
j = [j for j in range(initialFrame,finalFrame,frameStep)]

for i in range(initialFrame,finalFrame,frameStep):
    import_frames(i)


print(velocityAll)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_frames, j)