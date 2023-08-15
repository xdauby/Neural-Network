import random
import math
import numpy as np
import pandas as pd

def generate_regression_dataset(n : int = 100, 
                                start_range : float = -2.5, 
                                end_range : float = -2.5,
                                data_path : str = './src/datasets/regression_data.csv',
                                labels_path : str = './src/datasets/regression_labels.csv') -> None:
    #Simulate regression Y = 2 + X + 2*X^2
    step = 5/n
    X = np.arange(start_range, end_range, step)
    Y = 2 + 2*X*X + np.random.normal(0,0.5, size = n)

    df_labels = pd.DataFrame(np.array([Y]).T)
    df_data = pd.DataFrame(np.array([np.ones(n), X, X*X]).T)

    df_labels.to_csv(labels_path, index=False, header=False)
    df_data.to_csv(data_path, index=False, header=False)

def generate_XO_dataset(n : int = 200,
                        data_path : str = './src/datasets/classification_data.csv',
                        labels_path : str = './src/datasets/classification_labels.csv') -> None:

    df_data = pd.DataFrame(columns=['pixel_' + str(i) for i in range(10*10)])
    df_labels = pd.DataFrame(columns=['labels'])

    #generate O
    for nsample in range(int(n/2)):
    
        sample = np.zeros((10,10))
        # center and radius of the circle (x, y)
        circle_x = 5
        circle_y = 5
        circle_radius = 3

        for point in range(50):
            # play with randomness
            theta = 2 * math.pi * random.random()
            random_radius = 2 * random.random()
            if 1.2 > random_radius > 0.8:
                r = circle_radius * random_radius
            else:
                r = circle_radius
            # calculating coordinates
            x = int(r * math.cos(theta) + circle_x)
            y = int(r * math.sin(theta) + circle_y)
            # assign coordinates
            if 0 <= x <= 9 and 0 <= y <= 9:
                sample[x][y] = 1
        df_data.loc[len(df_data)] = sample.flatten()
        df_labels.loc[len(df_data)] = [0]
        
    #generate X
    for nsample in range(int(n/2)):
        sample = np.zeros((10,10))
        for point in range(35):
            # play with randomness
            line_direction = random.random()
            position = np.random.randint(1,9)
            epsilon = 1 - np.where(np.random.multinomial(1, [0.125, 0.75, 0.125]) == 1)[0][0]
            # calculating coordinates
            if line_direction < 0.5:
                position_x = position
                position_y = position + epsilon
                sample[position_x][position_y] = 1 
            else :
                position_x = position
                position_y = 9 - position_x + epsilon
                sample[position_x][position_y] = 1
        
        df_data.loc[len(df_data)] = sample.flatten()
        df_labels.loc[len(df_data)] = [1]

    df_labels.to_csv(labels_path, index=False, header=False)
    df_data.to_csv(data_path, index=False, header=False)
    
        
generate_XO_dataset()