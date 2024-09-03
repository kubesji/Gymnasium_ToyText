# Gymnasium: Toy text
Solutions for toy text environments of Gymnasium package using tabular methods.

Be caution with number of episodes and exploration when using tabular methods. If exploration to exploitation ratio decays quickly, some states might not be fully explored and agent mey stuck in there.

## Frozen lake
In the non-slippery mode, trained agent find a way in 100% case as expected. In the slippery mode, agent find a way in 50-60% cases.

## Cliff walking
There are no slippery surfaces or holes to terminate episode without reaching goal. Therefore, smaller amount of episodes is needed. Once the agent randomly finds the goal, it propagates rather quickly. Do not push number of episodes and epsilon decay too low as agent might not fully explore all states.

## Blackjack
Blackjack doesn't have too deep game tree, usually one or two actions. Therefore, we can only explore with no exploitation and still learn quickly.

As with most casino type games, odds favor dealer. This environment reshuffles card every game, thus the game reduced to game of chance, no cards counting possible. This leads to dealer winning roughly 50% games, 40% games won by agent and rest ending in draw.

## Taxi
This is the most complex task with the biggest game tree which naturally makes training time the longest. Result is agent that has 100% success rate and average of 13 steps per ride.