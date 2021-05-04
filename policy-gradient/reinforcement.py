import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from network import PolicyGradientNetwork


class Agent:
    def __init__(self,alpha=0.001,gamma=0.99,n_actions=4,fc1_dim=256,fc2_dim=256):
        self.alpha=alpha
        self.gamma=gamma
        self.n_actions=n_actions
        self.state_memory=[]
        self.action_memory=[]
        self.reward_memory=[]
        self.policy=PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.alpha))
    
    def choose_action(self,obs):
        state=tf.convert_to_tensor([obs],dtype=tf.float32)
        probs=self.policy(state)
        action_probs=tfp.distributions.Categorical(probs=probs)
        action=action_probs.sample()
        return action.numpy()[0]

    def store_transition(self,obs,action,reward):
        self.state_memory.append(obs)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    def learn(self):
        actions=tf.convert_to_tensor(self.action_memory,dtype=tf.float32)
        rewards=tf.convert_to_tensor(self.reward_memory,dtype=tf.float32)

        G=np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum=0
            discount=1
            for k in range(t,len(rewards)):
                G_sum+=rewards[k]*discount
                discount*=self.gamma
            G[t]=G_sum

        with tf.GradientTape() as tape:
            loss=0
            for idx,(g,state) in enumerate(zip(G,self.state_memory)):
                state=tf.convert_to_tensor([state],dtype=tf.float32)
                probs=self.policy(state)
                action_probs=tfp.distributions.Categorical(probs=probs)
                log_prob=action_probs.log_prob(actions[idx])
                loss+=-g*tf.squeeze(log_prob)

        gradient=tape.gradient(loss,self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient,self.policy.trainable_variables))

        self.state_memory=[]
        self.action_memory=[]
        self.reward_memory=[]
        