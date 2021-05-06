import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import ActorCriticNetwork

class Agent:
    def __init__(self,alpha=0.001,gamma=0.99,n_actions=2):
        self.alpha=alpha
        self.gamma=gamma
        self.n_actions=n_actions
        self.action=None
        self.action_space=[i for i  in range(0,n_actions)]
        self.actor_critic=ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=self.alpha))


    def choose_action(self,obs):
        state=tf.convert_to_tensor([obs])
        _,probs=self.actor_critic(state)
        action_probs=tfp.distributions.Categorical(probs=probs)
        action=action_probs.sample()
        self.action=action
        return action.numpy()[0]


    def save_model(self):
        print('Saving Models.....')
        self.actor_critic.save_weights(self.actor_critic.cpt_file)

    def load_model(self):
        print('Loading Model....')
        self.actor_critic.load_weights(self.actor_critic.cpt_file)

    def learn(self,state,reward,new_state,done):
        state=tf.convert_to_tensor([state],dtype=tf.float32)
        new_state=tf.convert_to_tensor([new_state],dtype=tf.float32)
        reward=tf.convert_to_tensor(reward,dtype=tf.float32)


        with tf.GradientTape() as tape:
            state_value,probs=self.actor_critic(state)
            new_state_value,_=self.actor_critic(new_state)
            state_value=tf.squeeze(state_value)
            new_state_value=tf.squeeze(new_state_value)
            action_probs=tfp.distributions.Categorical(probs=probs)
            log_prob=action_probs.log_prob(self.action)
            delta=(reward+self.gamma*(1-int(done))*new_state_value)-state_value
            actor_loss=-log_prob*delta
            critic_loss=delta**2
            loss=actor_loss+critic_loss

        gradient=tape.gradient(loss,self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients((grad, var) for (grad,var) in zip(gradient,self.actor_critic.trainable_variables) if grad is not None)