# https://github.com/fellowship/space-bandits/blob/master/toy_problem.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random, randint
from space_bandits import LinearBandits
from space_bandits import NeuralBandits
from space_bandits import load_linear_model
from space_bandits import load_neural_model
from time import time


#####################
# Generate Data
#####################

def get_customer(ctype=None):
    """Customers come from two feature distributions.
    Class 1: mean age 25, var 5 years, min age 18
             mean ARPU 100, var 15
    Class 2: mean age 45, var 6 years
             mean ARPU 50, var 25
    """
    if ctype is None:
        if random() > .5: #coin toss
            ctype = 1
        else:
            ctype = 2
    age = 0
    ft = -1
    if ctype == 1:
        while age < 18:
            age = np.random.normal(25, 5)
        while ft < 0:
            ft = np.random.normal(100, 15)
    if ctype == 2:
        while age < 18:
            age = np.random.normal(45, 6)
        while ft < 0:
            ft = np.random.normal(50, 25)
    age = round(age)
    return ctype, (age, ft)

def get_rewards(customer):
    """
    There are three actions:
    promo 1: low value. 10 dollar if accept
    promo 2: mid value. 25 dollar if accept
    promo 3: high value. 100 dollar if accept
    
    Both groups are unlikely to accept promo 2.
    Group 1 is more likely to accept promo 1.
    Group 2 is slightly more likely to accept promo 3.
    
    The optimal choice for group 1 is promo 1; 90% acceptance for
    an expected reward of 9 dollars each.
    Group 2 accepts with 25% rate for expected 2.5 dollar reward
    
    The optimal choice for group 2 is promo 3; 20% acceptance for an expected
    reward of 20 dollars each.
    Group 1 accepts with 2% for expected reward of 2 dollars.
    
    The least optimal choice in all cases is promo 2; 10% acceptance rate for both groups
    for an expected reward of 2.5 dollars.
    """
    if customer[0] == 1: #group 1 customer
        if random() > .1:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .90:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .98:
            reward3 = 100
        else:
            reward3 = 0
    if customer[0] == 2: #group 2 customer
        if random() > .75:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .90:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .80:
            reward3 = 100
        else:
            reward3 = 0
    return np.array([reward1, reward2, reward3])

def get_cust_reward():
    """returns a customer and reward vector"""
    cust = get_customer()
    reward = get_rewards(cust)
    age = cust[1]
    return np.array([age])/100, reward


get_customer()
get_rewards(get_customer())
get_cust_reward()


###########################
# Visualizing the Groups
###########################

plt.close('all')
group1 = [get_customer(ctype=1)[1] for x in range(1000)]
group2 = [get_customer(ctype=2)[1] for x in range(1000)]
plt.scatter([x[0] for x in group1], [x[1] for x in group1], label='group1');
plt.scatter([x[0] for x in group2], [x[1] for x in group2], label='group2');
plt.legend();
plt.show()


#################################
# Sanity Check: Expected Rewards
#################################

customers = [get_customer(ctype=1) for x in range(100000)]
rewards = np.concatenate([np.expand_dims(get_rewards(cust), axis=0) for cust in customers])
print('group 1 expected rewards: (100000 samples)', rewards.mean(axis=0))

customers = [get_customer(ctype=2) for x in range(100000)]
rewards = np.concatenate([np.expand_dims(get_rewards(cust), axis=0) for cust in customers])
rewards.mean(axis=0)
print('group 2 expected rewards: (100000 samples)', rewards.mean(axis=0))


#######################
# Linear model
#######################

num_actions = 3
num_features = 2

linear_model = LinearBandits(num_actions, num_features, initial_pulls=100)
optimal_choices = [None, 0, 2]


def iterate_model(model, optimal_choices, steps, records=None, plot_frequency=250, avg_length=150):
    """Goes through online learning simulation with model."""
    #these will track values for plotting
    if records is None:
        records = dict()
        records['timesteps'] = []
        records['c_reward'] = []
        records['cumulative_reward'] = 0
        records['m_reward'] = []
        records['maximum_reward'] = 0
        records['regret_record'] = []
        records['avg_regret'] = []
        start = 0
    else:
        start = records['timesteps'][-1]
    for i in range(start, start+steps):
        records['timesteps'].append(i)
        #generate a customer
        cust = get_customer()
        #generate customer decisions based on group
        reward_vec = get_rewards(cust)
        #prepare features for model
        context = np.array([cust[1]])
        best_choice = optimal_choices[cust[0]]
        #get reward for 'best' choice
        mx = reward_vec[best_choice]
        records['maximum_reward'] += mx
        records['m_reward'].append(records['maximum_reward'])
        action = model.action(context)
        #get reward for the action chosen by model
        reward = reward_vec[action]
        #regret is the opportunity cost of not choosing the optimal promotion
        regret = mx - reward
        records['regret_record'].append(regret)
        records['cumulative_reward'] += reward
        records['c_reward'].append(records['cumulative_reward'])
        model.update(context, action, reward)
        #plot occasionally
        if i <= avg_length:
            if i < avg_length:
                moving_avg=0
            else:
                moving_avg = np.array(records['regret_record']).mean()
            if i == avg_length:
                records['avg_regret'] = [moving_avg] * avg_length
        else:
            moving_avg = sum(records['regret_record'][-avg_length:])/avg_length
        records['avg_regret'].append(moving_avg)
        if i % plot_frequency == 0 and i > 0:
            c_rewardplt = np.array(records['c_reward'])/max(records['c_reward'])
            m_rewardplt = np.array(records['m_reward'])/max(records['m_reward'])
            regretplt = np.array(records['avg_regret'])/max(records['avg_regret'])
            plt.plot(records['timesteps'], c_rewardplt, label='cumulative reward')
            plt.plot(records['timesteps'], m_rewardplt, label='maximum reward')
            plt.plot(records['timesteps'], regretplt, color='red', label='mean regret')
            plt.title('Normalized Reward & Regret')
            plt.legend()
            plt.show()
    return records

plt.close('all')
records = iterate_model(linear_model, optimal_choices, 401, plot_frequency=400)



#######################
# Saving/Loading
#######################

#test linear model saving/loading
linear_model.save('test_path.pkl')
linear_model = load_linear_model('test_path.pkl')

#continue training
plt.close('all')
records = iterate_model(linear_model, optimal_choices, 401, plot_frequency=800, records=records)



######################################
# Visualizing the Decision Boundary
######################################

def plot_decision_boundary(model, X, Y, h=1, scale=1., parallelize=True, title='decision boundary', thompson=False, classic=False, n_threads=-1, flip_colors=True):
    ftnames = X.columns[0], X.columns[1]
    X = X.values
    #model.fit(X[:, :2], Y)
    x_min = X[:, 1].min() - .5
    x_max = X[:, 1].max() + .5
    y_min = X[:, 0].min() - .5
    y_max = X[:, 0].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    if classic:
        Z = model.classic_predict(np.c_[xx.ravel(), yy.ravel()]/scale, thompson=thompson)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]/scale, thompson=thompson, parallelize=parallelize)
    # Put the result into a color plot.
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, alpha=.25)
    # Add the training points to the plot.
    if flip_colors:
        Y = np.where(np.array(Y)==1, 0, 1)
    plt.scatter(X[:, 1], X[:, 0], c=Y, alpha=.5);
    #plt.scatter(X[:, 1], X[:, 0], c='black', alpha=.1);
    plt.xlabel(ftnames[1])
    plt.ylabel(ftnames[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max());
    plt.title(title)
    plt.show()
    
    

X = group1 + group2
Y = [1 for x in range(1000)] + [0 for x in range(1000)]
ages = [x[0] for x in X]
ARPUs = [x[1] for x in X]
as_df = pd.DataFrame()
as_df['ARPU'] = ARPUs
as_df['age'] = ages
X = as_df


plt.close('all')
plot_decision_boundary(linear_model, X, Y, h=.5, thompson=False, title='Decision Boundary Without Thompson Sampling')

t1 = time()

plt.close()
plot_decision_boundary(linear_model, X, Y, h=.5, thompson=True, parallelize=True, n_threads=3, title='Decision Boundary With Thompson Sampling')
print('took ', time()-t1)



#######################
# Neural Network
#######################

num_actions = 3
num_features = 2
memory_size = 10000

neural_model = NeuralBandits(num_actions, num_features, initial_pulls=100, memory_size=memory_size, layer_sizes=[50, 12])
assert neural_model.data_h.memory_size == memory_size

records = iterate_model(neural_model, optimal_choices, 401, plot_frequency=400)




#######################
# Saving Neural Models
#######################

neural_model.save('test_model')

neural_model = load_neural_model('test_model.zip')

plt.close('all')
records = iterate_model(neural_model, optimal_choices, 401, plot_frequency=800, records=records)


plt.close('all')
plot_decision_boundary(neural_model, X, Y, h=.5, thompson=False, title='Decision Boundary Without Thompson Sampling', n_threads=3)


t1 = time()
plt.close('all')
plot_decision_boundary(neural_model, X, Y, h=.5, thompson=True, parallelize=True, title='Decision Boundary With Thompson Sampling', n_threads=3)
print('took ', time() - t1)



#################################
# Training from Historic Data
#################################

def generate_dataframe(n_rows):
    df = pd.DataFrame()
    ages = []
    ARPUs = []
    actions = []
    rewards = []
    for i in range(n_rows):
        cust = get_customer()
        reward_vec = get_rewards(cust)
        context = np.array([cust[1]])
        ages.append(context[0, 0])
        ARPUs.append(context[0, 1])
        action = np.random.randint(0,3)
        actions.append(action)
        reward = reward_vec[action]
        rewards.append(reward)
    df['age'] = ages
    df['ARPU'] = ARPUs
    df['action'] = actions
    df['reward'] = rewards
    return df

df = generate_dataframe(2000)

df.head()


#################################
# Preparing Data for Training
#################################

#split data into triplets
contexts = df[['age', 'ARPU']]
actions = df['action']
rewards = df['reward']

#initialize new neural model
new_model = NeuralBandits(3, 2, layer_sizes=[50, 12], verbose=False)
#call .fit method; num_updates will repeat training n times
new_model.fit(contexts, actions, rewards, num_updates=10)


plt.close('all')
plot_decision_boundary(new_model, X, Y, h=.5, thompson=False, scale=1., title='Decision Boundary Without Thompson Sampling')

plt.close('all')
plot_decision_boundary(new_model, X, Y, h=.5, thompson=True, scale=1., title='Decision Boundary With Thompson Sampling', n_threads=3)


#################################
# Biased Data
#################################

def generate_biased_dataframe(n_rows):
    df = pd.DataFrame()
    ages = []
    ARPUs = []
    actions = []
    rewards = []
    for i in range(n_rows):
        cust = get_customer()
        reward_vec = get_rewards(cust)
        context = np.array([cust[1]])
        age = context[0, 0]
        ARPU = context[0, 1]
        ages.append(age)
        ARPUs.append(ARPU)
        if ARPU <= 50:
            action = 0
        elif ARPU <= 100:
            action = 1
        else:
            action = 2
        actions.append(action)
        reward = reward_vec[action]
        rewards.append(reward)
    df['age'] = ages
    df['ARPU'] = ARPUs
    df['action'] = actions
    df['reward'] = rewards
    return df

df = generate_biased_dataframe(2000)

contexts = df[['age', 'ARPU']]
actions = df['action']
rewards = df['reward']

bias_model = NeuralBandits(3, 2, layer_sizes=[50, 12], verbose=False)
bias_model.fit(contexts, actions, rewards, num_updates=10)

plt.close('all')
plot_decision_boundary(bias_model, X, Y, h=.5, thompson=False, scale=1., title='Decision Boundary Without Thompson Sampling')

plt.close('all')
plot_decision_boundary(bias_model, X, Y, h=.5, thompson=True, scale=1., title='Decision Boundary With Thompson Sampling', n_threads=3)


#################################
# Nonlinear problem
#################################

# Generate Data

def get_customer(ctype=None):
    """Customers come from two feature distributions.
    Class 1: mean age 25, var 5 years, min age 18
             mean ARPU 100, var 15
    Class 2: mean age 45, var 6 years
             mean ARPU 50, var 25
    """
    if ctype is None:
        ctype = randint(0,2)
    age = 0
    ft = -1
    if ctype == 0:
        while age < 18:
            age = np.random.normal(25, 5)
            ft = 125 - .1*(age-25)*(age-25) + np.random.normal(0, 4)
    if ctype == 1:
        while age < 18:
            age = np.random.normal(35, 2)
        while ft < 0:
            ft = np.random.normal(75, 3)
    if ctype == 2:
        while age < 18:
            age = np.random.normal(45, 6)
            ft = 25 + .25*(age-45)*(age-45) + np.random.normal(0, 4)
    age = round(age)
    return ctype, (age, ft)



def get_rewards(customer):
    """
    There are three actions:
    promo 1: low value. 10 dollar if accept
    promo 2: mid value. 25 dollar if accept
    promo 3: high value. 100 dollar if accept
    
    Expected Value Matrix:
    
           group1|group2|group3
    ----------------------------
    promo1|  $9  |  $1  |  $1
    ----------------------------
    promo2| $2.5 | $12.5| $1.25
    ----------------------------
    promo3| $1   |  $5  | $25
    
    We can see each group has an optimal choice.
    """
    if customer[0] == 0: #group 1 customer
        if random() > .1:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .90:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .99:
            reward3 = 100
        else:
            reward3 = 0
    if customer[0] == 1: #group 2 customer
        if random() > .9:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .50:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .95:
            reward3 = 100
        else:
            reward3 = 0
    if customer[0] == 2: #group 3 customer
        if random() > .9:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .95:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .75:
            reward3 = 100
        else:
            reward3 = 0
    return np.array([reward1, reward2, reward3])


def get_cust_reward():
    """returns a customer and reward vector"""
    cust = get_customer()
    reward = get_rewards(cust)
    age = cust[1]
    return np.array([age]), reward



customers = [get_customer(ctype=0) for x in range(100000)]
rewards = np.concatenate([np.expand_dims(get_rewards(cust), axis=0) for cust in customers])
print('group 1 expected rewards: (10000 samples)', rewards.mean(axis=0))

customers = [get_customer(ctype=1) for x in range(100000)]
rewards = np.concatenate([np.expand_dims(get_rewards(cust), axis=0) for cust in customers])
print('group 1 expected rewards: (10000 samples)', rewards.mean(axis=0))

customers = [get_customer(ctype=2) for x in range(100000)]
rewards = np.concatenate([np.expand_dims(get_rewards(cust), axis=0) for cust in customers])
rewards.mean(axis=0)
print('group 2 expected rewards: (10000 samples)', rewards.mean(axis=0))

optimal_choices = [0, 1, 2]
#confirm expected rewards


group1 = []
group2 = []
group3 = []
for i in range(1000):
    group1.append(get_customer(0))
    group2.append(get_customer(1))
    group3.append(get_customer(2))

plt.close('all')    
plt.scatter([x[1][0] for x in group1], [x[1][1] for x in group1], label='group1')
plt.scatter([x[1][0] for x in group2], [x[1][1] for x in group2], label='group2')
plt.scatter([x[1][0] for x in group3], [x[1][1] for x in group3], label='group3')
plt.xlabel('age')
plt.xlabel('ARPU')
plt.title('Another Distribution Example')
plt.legend();
plt.show()



num_actions = 3
num_features = 2

linear_model = LinearBandits(num_actions, num_features, initial_pulls=100)

plt.close('all')
records = iterate_model(linear_model, optimal_choices, 401, plot_frequency=400)


X = group1 + group2 + group3
X = [x[1] for x in X]
Y = [0 for x in range(1000)] + [1 for x in range(1000)] + [2 for x in range(1000)]
ages = [x[0] for x in X]
ARPUs = [x[1] for x in X]
as_df = pd.DataFrame()
as_df['ARPU'] = ARPUs
as_df['age'] = ages
X = as_df


plt.close('all')
plot_decision_boundary(linear_model, X, Y, h=.5, thompson=False, scale=1., title='Decision Boundary Without Thompson Sampling', flip_colors=False)


plt.close('all')
plot_decision_boundary(linear_model, X, Y, h=.5, thompson=True, scale=1., title='Decision Boundary With Thompson Sampling', flip_colors=False)



neural_model = NeuralBandits(num_actions, num_features, layer_sizes=[50,12], initial_pulls=100, verbose=False)
records = iterate_model(neural_model, optimal_choices, 3000, plot_frequency=3000)

plt.close('all')
plot_decision_boundary(neural_model, X, Y, h=.5, thompson=False, scale=1., title='Decision Boundary Without Thompson Sampling', flip_colors=False)

plt.close('all')
plot_decision_boundary(neural_model, X, Y, h=.5, thompson=True, scale=1., title='Decision Boundary With Thompson Sampling', flip_colors=False)

