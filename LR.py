import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


satellite_dict={
    'Aqua': 1,
    'Terra': 0
}

day_night_dict={
    'D': 1,
    'N': 0
}

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    
    # Min-Max Scalar
    def __init__(self):
        self.mins=None
        self.maxes=None
        
    def __call__(self,features, is_train=False):
        
        if is_train:
            
            #Min-max normalization
            self.mins= features.min(axis=0)
            self.maxes = features.max(axis=0)

        scaled_features = (features-self.mins)/(self.maxes-self.mins)
        
        return scaled_features

def get_correlation_plots():
    df=pd.read_csv("data/train.csv", index_col=0)

    df["satellite"]=df["satellite"].map(satellite_dict)
    df["daynight"]=df["daynight"].map(day_night_dict)

    #Processing acq_date
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    months = df['acq_date'].dt.month
    day_of_month = df['acq_date'].dt.day

    df=df.drop([ "version","instrument", "acq_date"],axis=1)

    
    ######
    # Uncomment this code to use Basis functions
    #df['bright_t31**9'] = df['bright_t31']**9
    #df['confidence**6'] = df['confidence']**6
    #df['X'] = np.cos(df['latitude']) * np.cos(df['longitude'])
    #df['Y'] = np.cos(df['latitude']) * np.sin(df['longitude'])
    #df['Z'] = np.sin(df['latitude'])
    ######

    df['sin(month)']=np.sin(months)
    df['day_of_month'] = day_of_month

    time = []
    for i in df['acq_time']:
        
        if i > 0 and  i <= 600:
            time.append(1)
        elif i > 600 and i<=1200:
            time.append(2)
        elif i > 1200 and i <= 1800:
            time.append(3)
        else:
            time.append(4)
    df['acq_time'] = time

    
    scaler=Scaler()
    df = pd.DataFrame(scaler(df, True))
    correlations = df.corr()
    (correlations['frp'].drop('frp').sort_values(ascending=False).plot.barh())
    plt.title('Correlations of frp')
    plt.xlabel('correlation')
    plt.ylabel('features')

    

def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    
    df=pd.read_csv(csv_path, index_col=0)
    
    #Convert categorical data to numeric
    df["satellite"]=df["satellite"].map(satellite_dict)
    df["daynight"]=df["daynight"].map(day_night_dict)
    
    dummies = pd.get_dummies(df[['acq_date']], drop_first=True)
    df = pd.concat([df.drop(['acq_date'],axis=1), dummies],axis=1)
    
    #Converting acq_time to group of 4
    time = []
    for i in df['acq_time']:
        
        if i > 0 and  i <= 600:
            time.append(1)
        elif i > 600 and i<=1200:
            time.append(2)
        elif i > 1200 and i <= 1800:
            time.append(3)
        else:
            time.append(4)
    df['acq_time'] = time
    
    
    ############ Basis Functions
    
    df['lat*long'] = df['latitude'] * df['longitude']
    df['bright_t31*bright'] = df['bright_t31'] * df['brightness']
    df['conf*track'] = df['confidence'] * df['track']
    df['satellite*scan'] = df['satellite'] * df['scan']
    
    df['bright_t31**9'] = df['bright_t31']**9
    df['confidence**6'] = df['confidence']**6
    
    df['bright*confid'] = df['bright_t31**9'] * df['confidence**6']
    df['scan*track']=df['scan']*df['track']
    
    df['mult_of_features'] = df['scan'] * df['track'] * df['bright_t31'] * df['brightness']
    df['mult_of_features_2'] = df['mult_of_features'] * df['latitude'] * df['longitude'] * df['confidence'] 
    
    df['acq_time*confidence'] = df['acq_time'] * df['confidence'] 
    df['acq_time*track'] = df['acq_time'] * df['track']

    ############
    
    dummies = pd.get_dummies(df[['acq_time']], drop_first=True)
    df = pd.concat([df.drop(['acq_time'],axis=1), dummies],axis=1)

    
    
    
    #Dropping unnecessary features
    df = df.drop(["version","instrument"], axis=1)
    
    #Remove frp column if present
    if('frp' in df.columns.to_list()):
        df = df.drop(['frp'], axis=1)
        
    #Convert to numpy
    feature_matrix=df.to_numpy()
    if scaler is not None:
        feature_matrix = scaler(feature_matrix, is_train)
    
    #Adding w0 term from y = wx + w0
    w0 = np.ones((feature_matrix.shape[0],1))
    feature_matrix = np.concatenate((w0, feature_matrix), axis=1)
    
    return feature_matrix

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    df=pd.read_csv(csv_path)
    
    return np.array(df['frp'])[:, None]
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''

    # W = inv(X.T*X + C*I) * X.T * Y
    feature_matrix_T = feature_matrix.T
    val = np.matmul(
        np.matmul(
            np.linalg.inv(
                np.matmul(
                feature_matrix_T,
                feature_matrix
                ) + C * np.identity(feature_matrix_T.shape[0])),
            feature_matrix_T
            ),
        targets
        )
    return val

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    
    # y = X*W
    return np.matmul(feature_matrix, weights)

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    
    #As we scaled targets logarithmically, scale targets and predictions exponentially to get original values
    #pred = np.exp(get_predictions(feature_matrix, weights))-1
    pred = get_predictions(feature_matrix, weights)

    
    #MSE=(1/N)*Summation((pred-targ)**2)
    return np.mean((pred - targets)**2)

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    
    return (weights**2).sum()

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    
    return mse_loss(feature_matrix, weights, targets) + C * l2_regularizer(weights)

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    pred = get_predictions(feature_matrix, weights)
    
    # partial derivative of Loss w.r.t weights
    # (-2/N) * (X*(Y-Predictions)) + 2CW
    return (-2/feature_matrix.shape[0]) * (np.matmul(feature_matrix.T, (targets - pred))) + 2*C*weights

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    indexes=np.random.choice(np.size(feature_matrix,0),batch_size,replace=False)
    feature_matrix_B = np.empty((0,np.size(feature_matrix,1)),feature_matrix.dtype)
    target_B=np.zeros(len(indexes))
    j=0  
    for i in indexes:
        feature_matrix_B = np.vstack([feature_matrix_B, feature_matrix[i]])
        target_B[j]=targets[i]
        j+=1

    return (feature_matrix_B,target_B[:,None])
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.zeros((n,1))

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    return value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    

    return weights - lr * gradients

def early_stopping(val_loss, weights, step):
    '''
    Description:
    check whether if current loss is smaller than previous loss. If small then save the weights and return True else return False
    '''

    '''
    Arguments:
    # val_loss: current validation loss
    # weights: current learned weights
    # step: current training step
    '''
    
    #If first step, save the weights and loss
    if step == 0:
        np.save('weights', weights)
        np.save('loss', val_loss)
        return True
    
    loss = np.load('loss.npy')
    
    #Compare with previous loss
    if val_loss < loss:
        np.save('weights', weights)
        np.save('loss', val_loss)
        return True
    
    return False
    

def plot_trainsize_losses():
    '''
    Description:
    plot losses on the development set instances as a function of training set size 
    '''


    scaler=Scaler()
    feature_matrix_train=get_features('data/train.csv',True,scaler)
    targets_train=get_targets('data/train.csv')
    weights_train=analytical_solution(feature_matrix_train,targets_train, C=1e-8)

    features_dev=get_features('data/dev.csv',False,scaler)
    targets_dev=get_targets('data/dev.csv')

    l=[]
    k=[]
    #for i in (5000,10000,15000,20000,feature_matrix_train.shape[0]):
    for i in range(1000, feature_matrix_train.shape[0], 1000):
        fm=feature_matrix_train[0:i,:]
        tg=targets_train[0:i,:]
        weights=analytical_solution(fm,tg)
        y=mse_loss(features_dev,weights,targets_dev)
        l.append(i)
        k.append(y)
        
    
    plt.plot(l,k,marker='o')
    plt.xlabel("size of the training set")
    plt.ylabel("mean-square error")
    plt.show()


def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows -- 
    '''
    
    weights = initialize_weights(train_feature_matrix.shape[1])
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    
    #First loss
    early_stopping(dev_loss, weights, 0)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
            
            #Early stopping
            is_good = early_stopping(dev_loss, weights, step)


        '''
        implement early stopping etc. to improve performance.
        '''

    #Early stoping weights
    return np.load('weights.npy')
    #Without early stopping
    #return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    #predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    
    #Compute correlation between features
    #get_correlation_plots()

    
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')
    
    a_solution = analytical_solution(train_features, train_targets, C=1e-5)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.001,
                        C=1e-5,
                        batch_size=16,
                        max_steps=60000,
                        eval_steps=1000)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    
    #plot_trainsize_losses()
