#!/usr/bin/env python

class ObserverAgent():
    def step(self, time_step, actions):
    	#print("info:" + str(info))
    	if (len(actions) != 0):
    		print("actions:" + str(actions))
        #print("{}".format(time_step.observation["game_loop"]))
