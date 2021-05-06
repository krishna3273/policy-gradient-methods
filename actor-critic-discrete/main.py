import gym
import numpy as np
from actor_critic import Agent
from utils import plotLearning


if __name__=='__main__':
    env=gym.make('CartPole-v0')
    agent=Agent(alpha=0.001,n_actions=env.action_space.n)
    n_episodes=400
    filename='cartpole.png'
    best_score=env.reward_range[0]
    score_history=[]
    load_checkpoint=False

    if load_checkpoint:
        agent.load_model()

    for i in range(n_episodes):
        obs=env.reset()
        done=False
        score=0
        while not done:
            action=agent.choose_action(obs)
            new_state,reward,done,info=env.step(action)
            score+=reward
            if not load_checkpoint:
                agent.learn(obs,reward,new_state,done)
            obs=new_state
        score_history.append(score)
        avg_score=np.mean(score_history[-100:])
        print(f'episode-{i},score={score},avg-score={avg_score}')
        if avg_score>best_score:
            best_score=avg_score
            if not load_checkpoint:
                agent.save_model()

    
    x=[i+1 for i in range(n_episodes)]
    plotLearning(score_history,filename=filename,x=x,window=100)