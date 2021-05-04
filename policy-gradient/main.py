import gym
import numpy as np
from reinforcement import Agent
from utils import plotLearning
import warnings
warnings.filterwarnings('ignore')

if __name__=="__main__":
    agent=Agent(alpha=0.0005,gamma=0.99,n_actions=4)
    env=gym.make('LunarLander-v2')
    score_history=[]
    n_episodes=2000

    for i in range(n_episodes):
        done=False
        score=0
        obs=env.reset()

        while not done:
            action=agent.choose_action(obs)
            new_state,reward,done,info=env.step(action)
            agent.store_transition(obs,action,reward)
            obs=new_state
            score+=reward
        score_history.append(score)

        agent.learn()

        avg_score=np.mean(score_history[-100:])

        print(f'episode-{i},score:{format(score,".1f")},avg_score:{avg_score}')

    filename="lunar-lander.png"
    plotLearning(score_history,filename,window=100)