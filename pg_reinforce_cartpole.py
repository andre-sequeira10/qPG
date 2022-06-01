from statistics import mean
import matplotlib.pyplot as plt
import gym

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import pennylane as qml 
from pennylane.templates import StronglyEntanglingLayers as SEL 
from pennylane.templates import BasicEntanglerLayers as BEL 
from pennylane.templates import IQPEmbedding
from pennylane.templates import AngleEmbedding
from pennylane import expval as expectation
from pennylane import PauliZ as Z 
from pennylane import PauliX as X 

from pennylane import numpy as np 
from torch.distributions import Categorical

import wandb

episodes=1000
n_layers = 4
n_qubits = 4    
lr_q = 0.1
batch_size = 10
basis_change=False 
ent="mod"
init="normal"
nm = "BORN-RX-layers-{}||lr-{}||entanglement-{}||basis_change-{}||init-{}||batch-{}||episodes-{}".format(n_layers,lr_q,ent,basis_change,init,batch_size,episodes)
'''
#wandb.init(name=nm,project="qPG")#, entity="quantumai")

wandb.config = {
  "learning_rate": lr_q,
  "epochs": 1000,
  "batch_size": batch_size,
  "layers": n_layers
}
'''
device = qml.device("default.qubit", wires = n_qubits)

def normalize(vector):
    norm = np.max(np.abs(np.asarray(vector)))
    return vector/norm
    
def ansatz(weights, n_layers=1, change_of_basis=False, entanglement="all2all"):

        for l in range(n_layers):
            if change_of_basis==True:
                for i in range(len(weights[l])):
                    qml.Rot(*weights[l][i],wires=i)
                    #qml.RY(weights[l][i][0],wires=i)
                    #qml.RZ(weights[l][i][1],wires=i)

            else:     
                for i in range(len(weights[l])):
                    qml.RY(weights[l][i][0],wires=i)
                    qml.RZ(weights[l][i][1],wires=i)

            if entanglement == "all2all":
                for q1 in range(n_qubits-1):    
                    for q2 in range(q1+1, n_qubits):
                        #qml.CNOT(wires=[q1,q2])
                        qml.CZ(wires=[q1,(q1+l+1)%n_qubits])

            
            elif entanglement == "mod":
                if not (l+1)%n_qubits:
                    l=0
                for q1 in range(n_qubits):
                    qml.CNOT(wires=[q1,(q1+l+1)%n_qubits])
                    #qml.CZ(wires=[q1,(q1+l+1)%n_qubits])

            elif entanglement == "linear":
                for q1 in range(n_qubits-1):    
                    qml.CNOT(wires=[q1,q1+1])

            elif entanglement == "circular":
                #if l+1 < n_layers:
                for q1 in range(n_qubits):
                    qml.CNOT(wires=[q1,(q1+1)%n_qubits])
                    #qml.CZ(wires=[q1,(q1+1)%n_qubits])

            
            elif entanglement == "nn":
                qml.CNOT(wires=[0,1])
                qml.CNOT(wires=[2,3])
                qml.CNOT(wires=[1,2])
        
#@qml.batch_input(argnum=0)
@qml.qnode(device, interface="torch", diff_method="backprop")
def qcircuit(inputs, weights0):
    
    #theta_0 = np.arccos(inputs[0])
    #theta_1 = np.arccos(inputs[2])
    #normalize between [-pi,pi] 
    #newvalue1= ((inputs[4] + 4*np.pi)/(8*np.pi)) * (2*np.pi) - np.pi
    #newvalue2= ((inputs[5] + 9*np.pi)/(18*np.pi)) * (2*np.pi) - np.pi
    #inpt = inputs[4:]
    inpts = normalize(inputs)
    #new_input = np.append(np.array([theta_0,theta_1]),inputs[4:])
    #normalized_input = np.array([theta_0,theta_1,newvalue1,newvalue2])
    #itan = np.arctan(inputs)    
    #AngleEmbedding(normalized_input[:2], wires=[0,1], rotation="X")
    #AngleEmbedding(normalized_input[2:], wires=[2,3], rotation="X")
    #AngleEmbedding(inputs[4:], wires=[2,3], rotation="X")
    AngleEmbedding(inpts, wires=range(n_qubits), rotation="X")
    ansatz(weights0,n_layers=n_layers, entanglement=ent)
    #SEL(weights0, wires=range(n_qubits))#, rotation=qml.RY)
    #ansatz(weights1, n_layers=1, change_of_basis=True, entanglement=None)

    #return [expectation(Z(0)), expectation(Z(1)), expectation(Z(2))]
    #return [expectation(Z(0)), expectation(Z(1))]# @ Z(1) @ Z(2))]
    return qml.probs(wires=0)# @ Z(1) @ Z(2))]


class policy_estimator_q(nn.Module):        
    def __init__(self, env):
        super(policy_estimator_q, self).__init__()
        #weight_shapes = {"weights0":(n_layers, n_qubits, 2)}#,"coeffs":(3)}#,"weights2":(n_layers,n_qubits,3),"weights3":(n_layers,n_qubits,3),"weights4":(n_layers,n_qubits,3)}#, "weights5":(1,n_qubits,3)}
        weight_shapes = {"weights0":(n_layers, n_qubits,2)}#,"weights1":(1,n_qubits,3)}#,"weights2":(n_layers,n_qubits,3),"weights3":(n_layers,n_qubits,3),"weights4":(n_layers,n_qubits,3)}#, "weights5":(1,n_qubits,3)}
        import functools

        self.uniform = functools.partial(torch.nn.init.uniform_, a=-np.pi, b=np.pi)
        self.glorot = functools.partial(torch.nn.init.normal_, mean=0.0, std=np.sqrt(1/3))
        self.normal = functools.partial(torch.nn.init.normal_, mean=0.0, std=1)
        #self.uniform_values = torch.nn.init.uniform_(weight_shapes["weights0"],a=min_value,b=max_value)
        #self.normal = torch.nn.init.normal_
        self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.normal)
        #self.fc1 = nn.Linear(3, 3)
        #self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        #self.ws = nn.Parameter(torch.ones(3), requires_grad=True)
    


    def forward(self, state):
        #QUANTUM ACTION SELECTION
        out = self.qlayer(torch.FloatTensor(state))
        #out=self.fc1(out)
        #out = torch.multiply(self.ws,out)
        #action_probs = F.softmax(self.beta*out, dim=-1)
        #probs = self.forward(state)
        m = Categorical(out)
        action = m.sample()
        return action.item(), m.log_prob(action)
        #action_probs = F.softmax(3*out, dim=-1)

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return (r - r.mean())

def reinforce(env, policy_estimator, num_episodes=600,
              batch_size=6, gamma=0.99, lr=0.01 , label=None):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_actions_tensor=[]
    batch_states = []
    batch_counter = 0
    
    LEARNING_RATE = lr

    for name,param in policy_estimator.named_parameters():
        print(name,"\n")
        print(param,"\n")

    optimizer = optim.Adam(policy_estimator.parameters(),
                            lr=LEARNING_RATE, 
                            amsgrad=True)
    #optimizer = optim.SGD(policy_estimator.parameters(),
                            #lr=LEARNING_RATE)
    grads = []

    import time 

    for ep in range(num_episodes):
        s_0 = env.reset()   
        states = []
        rewards = []
        actions = []
        log_actions = []
        complete = False
        while complete == False:
            #action_probs = policy_estimator.forward(s_0).detach().numpy()
            action, action_log_prob = policy_estimator.forward(s_0)
            log_actions.append(action_log_prob)
            #action_probs_sampler = torch.clone(action_probs).detach().numpy()


            #action = np.random.choice([-1,0,1], p=action_probs_sampler)
            

            #Cartpole and Mountaincar
            s_1, r, complete, _ = env.step(action)

            #rw = -(s_1[1] + np.sin(np.arcsin(s_1[1])+np.arcsin(s_1[3])))
            #Acrobot
            #s_1, r, complete, _ = env.step(action-1)
            
            states.append(s_0)
            
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_actions_tensor.extend(log_actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                mean_r = np.mean(total_rewards[-10:])
                
                # If batch is complete, update network
                if batch_counter == batch_size-1:
                    t_init = time.time()
                    optimizer.zero_grad()

                    #state_tensor = torch.FloatTensor(np.array(batch_states))
                    reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                    action_tensor = torch.LongTensor(np.array(batch_actions))

                    #outs = policy_estimator.forward(state_tensor)
                    #logprob = torch.log(outs)
                    logprob = torch.stack(batch_actions_tensor)

                    #print("logprob ",logprob)
                    #entropy2 = outs.entropy()
                    selected_logprobs = torch.multiply(reward_tensor,logprob)#[np.arange(len(action_tensor)), action_tensor]
                    #print("selected logprob", selected_logprobs)
                    loss = -torch.mean(selected_logprobs)
                    #loss = loss / batch_size 
                    print("mean - " , loss)

                    loss.backward()
                    optimizer.step()

                    t_end = time.time()

                    print("TIME - ", t_end-t_init)
                    
                    for name,param in policy_estimator.named_parameters():
                        if name == "ws":
                            print(name,"\n")
                            print(param,"\n")
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_actions_tensor=[]

                    batch_counter = 0
                    
                    grads_step = []
                    for param in policy_estimator.parameters():
                        grads_step.append(param.grad.view(-1))
                    
                    grads_step = torch.cat(grads_step).pow(2).numpy().mean()
                    
                    grads.append(grads_step)
                    #wandb.log({"grads": grads[-1]})

                mean_r = np.mean(total_rewards[-10:])

                #wandb.log({"total_rewards": total_rewards[-1]})
                #wandb.log({"mean_rewards_10": mean_r})

                # Optional
                #wandb.watch(policy_estimator)

                print("Ep: {} Average of last 10: {:.2f}".format(
                    ep + 1, mean_r))
                
    return total_rewards, grads

env = gym.make('CartPole-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('MountainCar-v0')

#env.seed(1)


pe_q= policy_estimator_q(env)
#model_q = torch.nn.DataParallel(pe_q)

rewards_q , grads_q= reinforce(env, pe_q , num_episodes=episodes, batch_size=batch_size, lr=lr_q, gamma=0.99)

for i in range(10):
    s0 = env.reset()
    complete = False
    while not complete:
        #action_probs = pe_q.forward(s0).detach().numpy()
        action, action_log_prob = pe_q.forward(s0)

                #action = np.random.choice(action_space, p=action_probs)
        #action = np.random.choice([-1,0,1], p=action_probs)
        s_1, r, complete, _ = env.step(action)
        env.render()
        s0 = s_1

