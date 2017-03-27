
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:

def normalizer(x):
    temp = np.zeros(len(x))
    sum = 0
    for i in range(len(x)):
        sum += x[i]
    for i in range(len(x)):
        temp[i]=x[i]/sum
    return temp

def random_row_generator(x, uniform = False):
    if uniform:
         return normalizer(np.ones(x))
    return normalizer(np.random.rand(x))

def random_sequence_generator(states, length):
    return np.random.randint(states, size = length)


# In[3]:

def print_setting(setting):
    print("Initial probability")
    print(setting['initial'])
    print("Transition matrix probability")
    print(setting['transition'])
    print("Observation matrix probability")
    print(setting['observation'])


# # Initial problem setting

# In[4]:

def HMM_initalization(observed_states, hidden_states, uniform =False):
    initial = random_row_generator(hidden_states,uniform)
    transition = []
    observation = []
    for i in range(hidden_states):
        transition.append(random_row_generator(hidden_states,uniform))
    for i in range(hidden_states):
        observation.append(random_row_generator(observed_states,uniform))
    transition = np.array(transition)
    observation = np.array(observation)
    return {'initial' : initial, 'transition' : transition, 'observation' : observation}


# In[5]:

hstates = 3
ostates = 4
setting = HMM_initalization(ostates, hstates)
print_setting(setting)


# # Evaluation Problem
#     Forward Recursion
#     Backwrd Recursion

#     Forward_k[i] = P(o1o2...ok, qk=si)  
#     Backward_k[i] = P(o(k+1)o(k+2)...oL|qk=si)

# In[6]:

observed_length = 10


# In[7]:

def encoding_problem(observed_sequence, initial, transition, observation):
    '''
    observed_sequence : 1D array 
                        each value(0 ~ observed_states-1)
    initial : 1D array(hidden_states)
                initial probability of hidden states
    transition : 2D array(hidden_states*hidden_states)
                transition probability of hidden_state to hidden state
    observation : 2D array(hidden_states*observed_states)    
                Observed probability of observed states for given hidden states
    '''
    hidden_states, observed_states = observation.shape
    length = len(observed_sequence)
    
    if hidden_states!=len(initial) or hidden_states!=transition.shape[0] or hidden_states!=transition.shape[1]: 
        raise SizeError('Wrong input')    
    
    for i in range(length):
        if observed_sequence[i]<observed_states:
            continue
        else:
            raise WrongInputError('Observed sequence wrong')

    forward = np.zeros((length,hidden_states))
    for j in range(hidden_states):
        forward[0][j] = initial[j]*observation[j][observed_sequence[0]]
    for i in range(1, length):
        for j in range(hidden_states):
            for k in range(hidden_states):
                forward[i][j] += forward[i-1][k]*transition[k][j]*observation[j][observed_sequence[i]]
                
    backward = np.zeros((length,hidden_states))
    for j in range(hidden_states):
        backward[length-1][j] = 1
    for i in range(1,length):
        for j in range(hidden_states):
            for k in range(hidden_states):
                backward[length-1-i][j] += transition[j][k]*observation[k][observed_sequence[length-i]]*backward[length-i][k]
    return {'forward' : forward, 'backward' : backward}


# In[8]:

sequence = random_sequence_generator(ostates,observed_length)

init = setting['initial']
transit = setting['transition']
observe = setting['observation']

result = encoding_problem(sequence, init, transit, observe)
forward = result['forward']
backward = result['backward']

observed_sequence_probability = 0
for i in range(hstates):
    observed_sequence_probability+=init[i]*backward[0][i]*observe[i][sequence[0]]
print("Observed_sequence")
print(sequence)
print("Observed_sequence_probability")
print(np.sum(forward[observed_length-1]))  
print("Observed_sequence_probability")
print(observed_sequence_probability)


# # Decoding problem
#     Viterbi Algorithm

#     viterbi_k[j]=max_(q1...q(k-1)){P(q1,q2,...,q(k_1),q_k=sj,o1o2...ok)}

# In[9]:

def decoding_problem(observed_sequence, initial, transition, observation):
    '''
    observed_sequence : 1D array 
                        each value(0 ~ observed_states-1)
    initial : 1D array(hidden_states)
                initial probability of hidden states
    transition : 2D array(hidden_states*hidden_states)
                transition probability of hidden_state to hidden state
    observation : 2D array(hidden_states*observed_states)    
                Observed probability of observed states for given hidden states
    '''  
    
    hidden_states, observed_states = observation.shape
    length = len(observed_sequence)
    
    if hidden_states==len(initial) and hidden_states==transition.shape[0] and hidden_states==transition.shape[1]: 
        print("Hidden states : {}, observed states : {}".format(hidden_states, observed_states))
    else:
        raise SizeError('Wrong input')
    
    for i in range(length):
        if observed_sequence[i]<observed_states:
            continue
        else:
            raise WrongInputError('Observed sequence wrong')
    length = len(observed_sequence)
    
    back_tracking_table = np.zeros((length,hidden_states),dtype=np.int32)
    viterbi = np.zeros((length,hidden_states))
    for j in range(hidden_states):
        viterbi[0][j] = initial[j]*observation[j][observed_sequence[0]]
        back_tracking_table[0][j] = j
    for i in range(1, length):
        for j in range(hidden_states):
            temp = np.zeros(hidden_states)
            for k in range(hidden_states):
                temp[k] = viterbi[i-1][k]*transition[k][j]*observation[j][observed_sequence[i]] 
            back_tracking_table[i][j] = np.argmax(temp)
            viterbi[i][j] = np.max(temp)
    hidden_sequence = np.zeros(length, dtype=np.int32)
    hidden_sequence[length-1] = np.argmax(viterbi[length-1])
    for i in range(1,length):
        hidden_sequence[length-1-i] = back_tracking_table[length-i][hidden_sequence[length-i]]
    return {'viterbi' : viterbi, 'hidden_sequence' : hidden_sequence}


# In[10]:

sequence = random_sequence_generator(ostates,observed_length)
print("Observed_sequence")
print(sequence)
result = decoding_problem(sequence, init, transit, observe)
print("Viterbi")
print(result['viterbi'])
print("Hidden Sequence")
print(result['hidden_sequence'])


# # Learning problem

# In[11]:

def random_choice(states, p):
    if states != len(p):
        raise WrongstatesError
    r = np.random.rand()*np.sum(p)
    for i in range(states):
        r-=p[i]
        if r<=0:
            return i


# In[12]:

def observed_sequence_generator(length, initial, transition, observation):   
    '''
    length : length of seqs
    initial : 1D array(hidden_states)
                initial probability of hidden states
    transition : 2D array(hidden_states*hidden_states)
                transition probability of hidden_state to hidden state
    observation : 2D array(hidden_states*observed_states)    
                Observed probability of observed states for given hidden states
    '''  
    hidden_states, observed_states = observation.shape
    
    if hidden_states!=len(initial) or hidden_states!=transition.shape[0] or hidden_states!=transition.shape[1]: 
        raise SizeError('Wrong input')

    hidden_seq = np.zeros(length, dtype=np.int32)
    observed_seq = np.zeros(length, dtype =np.int32)
    
    hidden_seq[0] = random_choice(hidden_states, p=initial)

    for i in range(1, length):
        hidden_seq[i] = random_choice(hidden_states, p = transition[hidden_seq[i-1]])
    
    for i in range(length):
        observed_seq[i] = random_choice(observed_states, p= observation[hidden_seq[i]])
    return observed_seq


# In[19]:

hstates = 3
ostates = 4
observed_length = 100
observed_num = 1000
answer_setting = HMM_initalization(ostates, hstates)
print_setting(answer_setting)


#     gamma_k[i]=P(qk = si| o1o2...oL)
#     epsilon_k[i,j]=P(qk = si, q(k+1)=sj | o1o2...oL)

# In[14]:

def Expectation(observed_sequence, initial, transition, observation):
    result = encoding_problem(observed_sequence, initial, transition, observation)
    forward = result['forward']
    backward = result['backward']

    hidden_states, observed_states = observation.shape
    length = len(observed_sequence)

    gamma = np.zeros((length, hidden_states))
    epsilon = np.zeros((length-1, hidden_states, hidden_states))
    
    for i in range(length):
        temp_sum = 0
        for j in range(hidden_states):
            temp_sum += forward[i][j]*backward[i][j]
        for j in range(hidden_states):
            gamma[i][j] = forward[i][j]*backward[i][j]/temp_sum
    
    for i in range(length-1):
        temp_sum = 0
        for j in range(hidden_states):
            for k in range(hidden_states):
                temp_sum += forward[i][j]*backward[i+1][k]*transition[j][k]*observation[k][observed_sequence[i+1]]
        for j in range(hidden_states):
            for k in range(hidden_states):
                epsilon[i][j][k] = forward[i][j]*backward[i+1][k]*transition[j][k]*observation[k][observed_sequence[i+1]]/temp_sum
    return {'gamma' : gamma, 'epsilon' : epsilon}


# In[15]:

def Maximization(observed_sequences, gamma, epsilon, observed_states, hidden_states):
    initial = np.zeros(hidden_states)
    transition = np.zeros((hidden_states,hidden_states))
    observation = np.zeros((hidden_states, observed_states))
    data_size, length = observed_sequences.shape
    
    for i in range(hidden_states):
        for d in range(data_size):
            initial[i] += gamma[d][0][i]
        initial[i]/=data_size
    
    for i in range(hidden_states):
        for j in range(hidden_states):
            temp1 = 0
            temp2 = 0
            for d in range(data_size):
                for l in range(length-1):
                    temp1 += epsilon[d][l][i][j]
                    temp2 += gamma[d][l][i]
            transition[i][j] = temp1/temp2
    
    for i in range(hidden_states):
        for j in range(observed_states):
            temp1 = 0
            temp2 = 0
            for d in range(data_size):
                for l in range(length):
                    if observed_sequences[d][l]==j:
                        temp1+=gamma[d][l][i]
                    temp2+=gamma[d][l][i]
            observation[i][j] = temp1/temp2
    return {'initial' : initial, 'transition' : transition, 'observation' : observation}


# In[22]:

observed_storage = []
for i in range(observed_num):
    observed_storage.append(observed_sequence_generator(observed_length, answer_setting['initial']
                                ,answer_setting['transition']
                                ,answer_setting['observation']))
observed_storage = np.array(observed_storage)
print(observed_storage.shape)

#print_setting(answer_setting)
#print(observed_storage)


# In[23]:

def probability(seq, initial, transition, observation):
    return np.sum(encoding_problem(seq, initial, transition, observation)['forward'][observed_length-1])


# In[28]:

print_setting(answer_setting)
Learning_setting = HMM_initalization(ostates, hstates)
print_setting(Learning_setting)

answer_probability = 0
for dataset in range(len(observed_storage)):
    answer_probability += math.log(probability(observed_storage[dataset]
                            ,answer_setting['initial']
                            ,answer_setting['transition']
                            ,answer_setting['observation']))
print("Challenge probability : {}".format(answer_probability))

iteration = 1000
Learn_probability = np.zeros(iteration)
for iter in range(iteration):
    gamma_set = []
    epsilon_set = []
    for dataset in range(len(observed_storage)):
        temp = Expectation(observed_storage[dataset]
                            ,Learning_setting['initial']
                            ,Learning_setting['transition']
                            ,Learning_setting['observation'])
        gamma_set.append(temp['gamma'])
        epsilon_set.append(temp['epsilon'])
    gamma_set = np.array(gamma_set)
    epsilon_set = np.array(epsilon_set)
    Learning_setting = Maximization(observed_storage, gamma_set, epsilon_set, ostates, hstates)
    
    for dataset in range(len(observed_storage)):
        Learn_probability[iter] += math.log(probability(observed_storage[dataset]
                                        ,Learning_setting['initial']
                                        ,Learning_setting['transition']
                                        ,Learning_setting['observation']))
print_setting(Learning_setting)

plt.plot(Learn_probability)
plt.axhline(y=answer_probability, xmin=0, xmax=iteration, linewidth=2, color = 'k')
plt.show()

