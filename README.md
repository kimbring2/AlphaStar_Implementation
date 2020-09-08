# Introduction
I am trying to implement AlphaStar based on supplementary material of DeepMind.

# Reference
1. Download replay file : https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api
```
python download_replays.py --key=<your key> --secret=<your secret key> --version=4.8.2 --replays_dir=<your path> --extract
```

2. Extracting observation, action from replay file : https://github.com/narhen/pysc2-replay

# Detailed information
I am writing explanation for code at Medium as series.

1. First tutorial : https://medium.com/@dohyeongkim/alphastar-implementation-serie-part1-606572ddba99

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
