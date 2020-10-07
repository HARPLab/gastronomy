
Baseline 1:
Create 101 blocks in simulation with different model parameters 
Sample random block and a random action skill until a pair succeeds.
Then run this action on the "actual" robot.
Determine if skill succeeded or failed.
Then run the same skill on all 100 blocks and update particles by summing similar numbers and dividing by total.
Shows the speed of our approach and generalizability.
Baseline should work well because of a lot of prior knowledge.

Baseline 2:
Original method without joint contingency networks.
aka When running a randomly sampled skill, only update the probabilities of particles that
are from the same skill.
Shows joint contingency is useful.

Baseline 3:
Baseline 1 without modeling either friction or mass distribution
How would the model based method work if you are missing parameters in the model. 


Metrics: 
Update distribution either 5/10 times. Then sample 100-1000 actions from the distribution and determine success probabilities
KL Divergence with Test block which has actual distribution 


30 blocks in training data
Sample random action based on the neural networks 
Execute action and observe if skill succeeded or failed
Compare closest skill in training data with the random action.
Update block particles distribution on whether the results were similar
Weight action probabilities based on the block particle probabilities
