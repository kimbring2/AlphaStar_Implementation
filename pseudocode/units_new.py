from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum


# pylint: disable=invalid-name
class Neutral(enum.IntEnum):
	"""Neutral units."""
	BattleStationMineralField = 182
	BattleStationMineralField750 = 183
	CarrionBird = 184
	CleaningBot = 185
	CollapsibleRockTower = 186
	CollapsibleRockTowerDebris = 187
	CollapsibleRockTowerDebrisRampLeft = 188
	CollapsibleRockTowerDebrisRampRight = 189
	CollapsibleRockTowerDiagonal = 190
	CollapsibleRockTowerPushUnit = 191
	CollapsibleRockTowerPushUnitRampLeft = 192
	CollapsibleRockTowerPushUnitRampRight = 193
	CollapsibleRockTowerRampLeft = 194
	CollapsibleRockTowerRampRight = 195
	CollapsibleTerranTower = 196
	CollapsibleTerranTowerDebris = 197
	CollapsibleTerranTowerDiagonal = 198
	CollapsibleTerranTowerPushUnit = 199
	CollapsibleTerranTowerPushUnitRampLeft = 200
	CollapsibleTerranTowerPushUnitRampRight = 201
	CollapsibleTerranTowerRampLeft = 202
	CollapsibleTerranTowerRampRight = 203
	Crabeetle = 204
	Debris2x2NonConjoined = 205
	DebrisRampLeft = 206
	DebrisRampRight = 207
	DestructibleBillboardTall = 208
	DestructibleCityDebris4x4 = 209
	DestructibleCityDebris6x6 = 210
	DestructibleCityDebrisHugeDiagonalBLUR = 211
	DestructibleDebris4x4 = 212
	DestructibleDebris6x6 = 213
	DestructibleDebrisRampDiagonalHugeBLUR = 214
	DestructibleDebrisRampDiagonalHugeULBR = 215
	DestructibleIce4x4 = 216
	DestructibleIce6x6 = 217
	DestructibleIceDiagonalHugeBLUR = 218
	DestructibleRampDiagonalHugeBLUR = 219
	DestructibleRampDiagonalHugeULBR = 220
	DestructibleRock6x6 = 221
	DestructibleRockEx14x4 = 222
	DestructibleRockEx16x6 = 223
	DestructibleRockEx1DiagonalHugeBLUR = 224
	DestructibleRockEx1DiagonalHugeULBR = 225
	DestructibleRockEx1HorizontalHuge = 226
	DestructibleRockEx1VerticalHuge = 227
	Dog = 228
	InhibitorZoneMedium = 229
	InhibitorZoneSmall = 230
	KarakFemale = 231
	LabBot = 232
	LabMineralField = 233
	LabMineralField750 = 234
	Lyote = 235
	MineralField = 236
	MineralField450 = 237
	MineralField750 = 238
	ProtossVespeneGeyser = 239
	PurifierMineralField = 240
	PurifierMineralField750 = 241
	PurifierRichMineralField = 242
	PurifierRichMineralField750 = 243
	PurifierVespeneGeyser = 244
	ReptileCrate = 245
	RichMineralField = 246
	RichMineralField750 = 247
	RichVespeneGeyser = 248
	Scantipede = 256
	ShakurasVespeneGeyser = 250
	SpacePlatformGeyser = 251
	UnbuildableBricksDestructible = 252
	UnbuildablePlatesDestructible = 253
	UnbuildableRocksDestructible = 254
	UtilityBot = 255
	VespeneGeyser = 249
	XelNagaDestructibleBlocker8NE = 256
	XelNagaDestructibleBlocker8SW = 256
	XelNagaTower = 256


class Protoss(enum.IntEnum):
	"""Protoss units."""
	Adept = 137
	AdeptPhaseShift = 138
	Archon = 139
	Assimilator = 140
	AssimilatorRich = 141
	Carrier = 142
	Colossus = 143
	CyberneticsCore = 144
	DarkShrine = 145
	DarkTemplar = 146
	Disruptor = 147
	DisruptorPhased = 148
	FleetBeacon = 149
	ForceField = 150
	Forge = 151
	Gateway = 152
	HighTemplar = 153
	Immortal = 154
	Interceptor = 155
	Mothership = 156
	MothershipCore = 157
	Nexus = 158
	Observer = 159
	ObserverSurveillanceMode = 160
	Oracle = 161
	Phoenix = 162
	PhotonCannon = 163
	Probe = 164
	Pylon = 165
	PylonOvercharged = 166
	RoboticsBay = 167
	RoboticsFacility = 168
	Sentry = 169
	ShieldBattery = 170
	Stalker = 171
	Stargate = 172
	StasisTrap = 173
	Tempest = 174
	TemplarArchive = 175
	TwilightCouncil = 176
	VoidRay = 177
	WarpGate = 178
	WarpPrism = 179
	WarpPrismPhasing = 180
	Zealot = 181


terran_building_list = ['Armory', 'AutoTurret', 'Barracks', 'BarracksFlying', 'BarracksReactor', 'BarracksTechLab', 
                           'Bunker', 'CommandCenter', 'CommandCenterFlying', 'EngineeringBay', 'Factory', 'FactoryFlying',
                           'FactoryReactor', 'FactoryTechLab', 'FusionCore', 'GhostAcademy', 'MissileTurret', 'OrbitalCommand',
                           'OrbitalCommandFlying', 'PlanetaryFortress', 'Reactor', 'Refinery', 'RefineryRich', 'SensorTower', 
                           'Starport', 'StarportFlying', 'StarportReactor', 'StarportTechLab', 'SupplyDepot', 'SupplyDepotLowered',
                           'TechLab']
terran_infantry_unit_list = ['Ghost', 'GhostAlternate', 'GhostNova', 'Marauder', 'Marine', 'Reaper', 'SCV']
terran_vehicle_unit_list = ['MULE', 'Hellion', 'Hellbat', 'SiegeTank', 'SiegeTankSieged', 'Thor', 'Cyclone'] 
terran_ship_unit_list = ['VikingAssault', 'VikingFighter', 'Banshee', 'Battlecruiser' , 'Liberator', 'LiberatorAG', 'Medivac', 'Raven'] 
terran_etc_unit_list = ['PointDefenseDrone', 'WidowMine', 'WidowMineBurrowed']
class Terran(enum.IntEnum):
	"""Terran units."""
	Armory = 1
	AutoTurret = 2
	Banshee = 3
	Barracks = 4
	BarracksFlying = 5
	BarracksReactor = 6
	BarracksTechLab = 7
	Battlecruiser = 8
	Bunker = 9
	CommandCenter = 10
	CommandCenterFlying = 11
	Cyclone = 12
	EngineeringBay = 13
	Factory = 14
	FactoryFlying = 15
	FactoryReactor = 16
	FactoryTechLab = 17
	FusionCore = 18
	Ghost = 19
	GhostAcademy = 20
	GhostAlternate = 21
	GhostNova = 22
	Hellion = 23
	Hellbat = 24
	KD8Charge = 25
	Liberator = 26
	LiberatorAG = 27
	MULE = 28
	Marauder = 29
	Marine = 30
	Medivac = 31
	MissileTurret = 32
	Nuke = 33
	OrbitalCommand = 34
	OrbitalCommandFlying = 35
	PlanetaryFortress = 36
	PointDefenseDrone = 37
	Raven = 38
	Reactor = 39
	Reaper = 40
	Refinery = 41
	RefineryRich = 42
	RepairDrone = 43
	SCV = 44
	SensorTower = 45
	SiegeTank = 46
	SiegeTankSieged = 47
	Starport = 48
	StarportFlying = 49
	StarportReactor = 50
	StarportTechLab = 51
	SupplyDepot = 52
	SupplyDepotLowered = 53
	TechLab = 54
	Thor = 55
	ThorHighImpactMode = 56
	VikingAssault = 57
	VikingFighter = 58
	WidowMine = 59
	WidowMineBurrowed = 60


class Zerg(enum.IntEnum):
	"""Zerg units."""
	Baneling = 61
	BanelingBurrowed = 62
	BanelingCocoon = 63
	BanelingNest = 64
	BroodLord = 65
	BroodLordCocoon = 66
	Broodling = 67
	BroodlingEscort = 68
	Changeling = 69
	ChangelingMarine = 70
	ChangelingMarineShield = 71
	ChangelingZealot = 72
	ChangelingZergling = 73
	ChangelingZerglingWings = 74
	Cocoon = 75
	Corruptor = 76
	CreepTumor = 77
	CreepTumorBurrowed = 78
	CreepTumorQueen = 79
	Drone = 80
	DroneBurrowed = 81
	EvolutionChamber = 82
	Extractor = 83
	ExtractorRich = 84
	GreaterSpire = 85
	Hatchery = 86
	Hive = 87
	Hydralisk = 88
	HydraliskBurrowed = 89
	HydraliskDen = 90
	InfestationPit = 91
	InfestedTerran = 92
	InfestedTerranBurrowed = 93
	InfestedTerranCocoon = 94
	Infestor = 95
	InfestorBurrowed = 96
	Lair = 97
	Larva = 98
	Locust = 99
	LocustFlying = 100
	Lurker = 101
	LurkerBurrowed = 102
	LurkerDen = 103
	LurkerCocoon = 104
	Mutalisk = 105
	NydusCanal = 106
	NydusNetwork = 107
	Overlord = 108
	OverlordTransport = 109
	OverlordTransportCocoon = 110
	Overseer = 111
	OverseerCocoon = 112
	OverseerOversightMode = 113
	ParasiticBombDummy = 114
	Queen = 115
	QueenBurrowed = 116
	Ravager = 117
	RavagerBurrowed = 118
	RavagerCocoon = 119
	Roach = 120
	RoachBurrowed = 121
	RoachWarren = 122
	SpawningPool = 123
	SpineCrawler = 124
	SpineCrawlerUprooted = 125
	Spire = 126
	SporeCrawler = 127
	SporeCrawlerUprooted = 128
	SwarmHost = 129
	SwarmHostBurrowed = 130
	Ultralisk = 131
	UltraliskBurrowed = 132
	UltraliskCavern = 133
	Viper = 134
	Zergling = 135
	ZerglingBurrowed = 136


def get_unit_type(race, name):
	#print("race: " + str(race))
	#print("name: " + str(name))

	if race == "Neutral":
		unit_caterogy = 'Unknown'

		try:
			return Neutral[name], unit_caterogy
		except ValueError:
			print("ValueError")
			pass  # Wrong race.
	elif race == "Protoss":
		try:
			return Protoss[name]
		except ValueError:
			print("ValueError")
			pass  # Wrong race.
	elif race == "Terran":
		if name in terran_building_list:
			unit_caterogy = 'Building'
		elif name in terran_infantry_unit_list:
			unit_caterogy = 'Infantry'
		elif name in terran_vehicle_unit_list:
			unit_caterogy = 'Vehicle'
		elif name in terran_ship_unit_list:
			unit_caterogy = 'Ship'
		else:
			unit_caterogy = 'Etc'

		try:
			return Terran[name], unit_caterogy
		except ValueError:
			print("ValueError")
			pass  # Wrong race.
	elif race == "Zerg":
		try:
			return Zerg[name]
		except ValueError:
			print("ValueError")
			pass  # Wrong race.