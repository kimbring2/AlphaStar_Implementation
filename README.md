# Introduction
I am trying to implement AlphaStar based on supplementary material of DeepMind.

# Reference
1. Download replay file(4.8.2 version file is needed): https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api
2. Extracting observation, action from replay file: https://github.com/narhen/pysc2-replay

# Version
1. Python3
2. PySC2 3.0.0

# Running test
To check you setting all your computer environment correctly. Run 'env_test.py' file in your terminal.
Then, screen of PySC2 will start and you can see some activation of two Terran rule-based agent. 

# Detailed information
I am writing explanation for code at Medium as series.

1. First tutorial: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part1-606572ddba99
2. Second tutorial: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part2-3edced5df00b
3. Third tutorial: https://medium.com/@dohyeongkim/alphastar-implementation-series-part3-d315d2ad5a3

# Extract information from replay file
The following is the process of getting build order, unit and cumulative score information from one replay file in my PC working environment.

You should replace source as path of the folder where the replay file stored. Race is 1:Terran, 2:Zerg, 3:Protoss, respectively.

```
from trajectory import get_random_trajectory
info_1, info_2 = get_random_trajectory(source='/media/kimbring2/Steam/StarCraftII/Replays/4.8.2.71663-20190123_035823-1/', home_race='Terran', away_race=['Terran', 'Zerg', 'Protoss'], replay_filter=3500)

print(info_1)
['SCV', 'SCV', 'SupplyDepot', 'SCV', 'SCV', 'SCV', 'Barracks', 'Refinery', 'SCV', 'SCV', 'SCV', 'Reaper', 'Reaper', 'CommandCenter', 'Refinery', 'SCV', 'Reaper', 'Factory', 'SCV', 'SupplyDepot', 'Reaper', 'Bunker', 'SCV', 'SCV', 'Hellion', 'SupplyDepot', 'Marine', 'SCV', 'SCV', 'SCV', 'TechLab', 'TechLab', 'SCV', 'SCV', 'SCV', 'SCV', 'SCV', 'CommandCenter', 'SCV', 'SiegeTank', 'SCV', 'Starport', 'EngineeringBay', 'SCV', 'SCV', 'SCV', 'SCV', 'SCV', 'Marine', 'SiegeTank']

print(info_2)
{'score': 6052, 'idle_production_time': 272, 'idle_worker_time': 38, 'total_value_units': 2275, 'total_value_structures': 1900, 'killed_value_units': 150, 'killed_value_structures': 0, 'collected_minerals': 4365, 'collected_vespene': 912, 'collection_rate_minerals': 1483, 'collection_rate_vespene': 335, 'spent_minerals': 4150, 'spent_vespene': 700}
```

# Run AlphaStar main file
I add basic code for running AlphaStar such as PySC2 environment, RL state&action part. Please run below code with another file of same folder.

[alphastar.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/alphastar.py)

You can check that PySC2 environment will be created and producing of data of each step.

# Preprocess observation of PySC2
AlphaStar need to preprosess observation of PySC2 for selection action. I add a 3 Encoder,Core network and preprocessed code for that.  
[network.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/network.py)
[upgrades_new.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/upgrades_new.py)
[utils.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/utils.py)

Please run below code with another file of same folder. You can check agent produce some value by using raw observation of PySC2

[alphastar.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/alphastar.py)
