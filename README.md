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

1. Tutorial about replay file: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part1-606572ddba99
2. Tutorial about agent class: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part2-3edced5df00b
3. Tutorial about encoder network: https://medium.com/@dohyeongkim/alphastar-implementation-series-part3-d315d2ad5a3
3. Tutorial about head network: https://dohyeongkim.medium.com/alphastar-implementation-series-part4-ee64bb93fe59

# Preprocess observation of PySC2
I am adding the necessary code for running alphaStar based on psuedocode of DeepMind paper

1. Extract information from Replay file: [trajectory.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/trajectory.py)
2. Encoder, Core, Head network: [network.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/network.py)
3. Preprocessing function : [utils.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/utils.py)
4. Edited unit inforamtion of PySC2: [units_new.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/units_new.py)
5. Edited upgrade inforamtion of PySC2: [upgrades_new.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/upgrades_new.py)

You can run AlphaStar using below command after putting all additional file with same folder.
```
$ python alphastar.py 
```
