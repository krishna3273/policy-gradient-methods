import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorCriticNetwork(keras.Model):
    def __init__(self,n_actions,fc1_dim=1024,fc2_dim=512,name='actor_critic',check_dir='./checkpoint/actor_critic'):
        super(ActorCriticNetwork,self).__init__()
        self.fc1_dim=fc1_dim
        self.fc2_dim=fc2_dim
        self.n_actions=n_actions
        self.model_name=name
        self.check_dir=check_dir
        self.cpt_file=os.path.join(self.check_dir,name+'_ac')

        self.fc1=Dense(self.fc1_dim,activation='relu')
        self.fc2=Dense(self.fc2_dim,activation='relu')
        self.value=Dense(1,activation=None)
        self.policy=Dense(n_actions,activation='softmax')

    def call(self,inp):
        out=self.fc1(inp)
        out=self.fc2(inp)

        out_value=self.value(out)
        out_policy=self.policy(out)

        return out_value,out_policy