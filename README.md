# HMM(Hidden markov model)

## Environment 
python = 3.6
matplotlib
numpy

## Background 

    hidden state : h
    observe state : o

1) Initial probability for hidden states : 1D array(h)
 
2) Transition between hidden states : 2D array(h*h)

3) Observation of observed states inferred from hidden states : 2D array(h*o)

hidden seqeuences : q1,q2,q3...
observe seqeuences : o1,o2,o3...

## 1. Evaluation problem
**Objective**  
    Caculate P(o1o2....)

**Can be calculated from 2 kinds of recursion**  
    Forward_k[i] = P(o1o2...ok, qk=si)  
    Backward_k[i] = P(o(k+1)o(k+2)...oL|qk=si)

## 2. Decoding problem   

**Objective**  
    Find most probabable hidden seqeuences(q1q2...) for observed sequences(o1o2.....)  
    *argmax(P(q1q2....|o1o2....))*

**Dynamic programming and backward tracking**  
    so_called viterbi algorithm  
    *viterbi_k[j]=max_(q1...q(k-1)){P(q1,q2,...,q(k_1),q_k=sj,o1o2...ok)}*    

## 3. Learning problem

**Objective**  
    number of hidden states, observed states are given  
    Many many observed sequences are given  
    
    Find the parameters  
    1) Initial probability for hidden states : 1D array(h)
    2) Transition between hidden states : 2D array(h*h)
    3) Observation of observed states inferred from hidden states : 2D array(h*o)
    such that maximizes sum{P(o1o2....)}
