# Introduction
I am trying to implement AlphaStar based on supplementary material of DeepMind.

# Reference
1. Download replay file(4.8.2 version file is needed): https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api
2. Extracting observation, action from replay file: https://github.com/narhen/pysc2-replay
3. Transfomer network of Tensorflow 2.0: https://www.tensorflow.org/tutorials/text/transformer
4. Resblock of Tensorflow 2.0: https://zhuanlan.zhihu.com/p/87047648

# Version
1. Python3
2. PySC2 3.0.0
3. Tensorflow 2.2.0

# Running test
## MoveToBeacon
First, let's test the sample code for MoveToBeacon environment which is the simplest environment in PySC2 using model which has same network structure as AlphaStar. First, place [run.py](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/run.py), [network.py](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/network.py) files in your working folder. Next, start training by using below command.

```
$ python run.py --workspace_path /media/kimbring2/Steam/Relational_DRL_New/ --train True --gpu True --save True
```

In the case of MoveToBeacon environment, as shown in the graph below, total reward will continue to decrease after reaching the maximum reward. Therefore, load a weight of well trained when testing. The weight is saved every 5 episodes under the Models folder in the specified workspace.

<img src="image/MoveToBeacon_A2C.png" width="800">

After the training is completed, change a weight file name of best training to model. Then, test using the following command.

```
$ python run.py --workspace_path /media/kimbring2/Steam/Relational_DRL_New/ --visualize True --load True
```

<img src="image/alphastar_beacon.gif" width="800">

If the accumulated reward is over 20 per episode, you can see the Marine follow the beacon well, as in the video above.

# Detailed information
I am writing explanation for code at Medium as series.

1. Tutorial about replay file: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part1-606572ddba99
2. Tutorial about agent class: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part2-3edced5df00b
3. Tutorial about encoder network: https://medium.com/@dohyeongkim/alphastar-implementation-series-part3-d315d2ad5a3
4. Tutorial about head network: https://dohyeongkim.medium.com/alphastar-implementation-series-part4-ee64bb93fe59
5. Tutorial about training network: https://dohyeongkim.medium.com/alphastar-implementation-series-part5-fd275bea68b5
6. Tensorflow 2.0 inplementation of FullyConv model: https://dohyeongkim.medium.com/alphastar-implementation-series-part6-4044e7efb1ce

# Preprocess observation of PySC2
I am adding the necessary code for running AlphaStar based on psuedocode of DeepMind paper

1. Extract information from Replay file: [trajectory.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/trajectory.py)
2. Encoder, Core, Head network: [network.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/network.py)
3. Preprocessing function : [utils.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/utils.py)
4. Edited unit inforamtion of PySC2: [units_new.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/units_new.py)
5. Edited upgrade inforamtion of PySC2: [upgrades_new.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/upgrades_new.py)

You can run AlphaStar application by running [alphastar.py file](https://github.com/kimbring2/AlphaStar_Implementation/blob/master/pseudocode/alphastar.py) after putting all additional file in same folder.
```
$ python alphastar.py 
```

# License
Apache License 2.0
