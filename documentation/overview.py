import itertools

self = Agent()
self.world_modeler = WorldModeler()

# different game enviornments
# same physics engine
for each_task in tasks:
    
    # tell the agent to have an open mind
    self.world_modeler.freeze_exising_weights()
    self.world_modeler.prune_and_compress()
    self.world_modeler.add_fresh_neurons()
    
    # tell the agent to reset it's reward-loop expectations
    self.reward_estimator   = RewardPredictor()   # doesn't think about the future
    self.value_approximator = ValueApproximator() # discounts future rewards
    self.decision_maker     = DecisionMaker()     # decides what to do
    
    # iterate the enviornment
    for index in itertools.count(0):
        # 
        # take an action based on current world_features
        # 
        compressed_state = self.world_modeler.world_features
        action = self.decision_maker.decide(compressed_state)
        new_uncompressed_state, reward, done, _ = env.step(action)
        
        # 
        # update value approximator (critic)
        # 
        # this handles the discount factor, cares about long-term overall value
        state_value = self.value_approximator.predict(compressed_state)
        self.value_approximator.improve_self(
            prediction=state_value,
            new_information=reward,
        )
        
        # 
        # update policy (actor)
        # 
        # any traditional update
        self.decision_maker.improve_self(
            state=compressed_state,
            action=action,
            reward=state_value,
            new_state=new_uncompressed_state,
        )
        
        # 
        # update reward estimator
        # 
        # used by the world modeler
        # no discounting, doesn't care about future rewards
        # only cares about past rewards if they help predict the present
        predicted_reward_distribution = self.reward_estimator.predict(compressed_state)
        self.reward_estimator.improve_self(
            state=compressed_state,
            prediction=predicted_reward_distribution,
            actuality=reward,
        )
        
        # 
        # update the understanding of the world
        # 
        # this step is similar to a VAE in terms of automatic feature compression
        new_world_features = self.world_modeler.generate_new_status(
            current_state=new_uncompressed_state,
            # this^ answers "what new information do I have"
            current_policy=self.decision_maker.policy,
            # this^ answers "what am I planning on doing next"
            # ex: different policies (randomized/deterministic, agressive/passive) change predicted world states
        )
        # loss/update
        self.world_modeler.improve_understanding(
            focus=self.reward_estimator.feature_weights,
            # this^ tries to estimate which features of the enviornment are important
            prediction=compressed_state,
            # this^ "what did I predict?" (saves processing power vs re-computing prediction)
            actuality=new_world_features,
            # this^ "what actually happened?"
        )
        # the loss is split between a few tasks
        # - traditional auto encoder loss
        # - predicting future world features
        # - providing features that accurately predict the reward
        # each of these produces a gradient, and has a prospective weight
        # the gradients are combined
        # the traditional-auto-encoder weight is based on the amount of uncertainty
        # which starts at 100%, and decreses with the accuracy of the reward predictions