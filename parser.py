import pandas as pd

class Team:
	def __init__(self, name, wins=0, losses=0, ties=0):
		self.wins = wins
		self.losses = losses
		self.ties = ties
		self.name = name
		self.games = []

class Parser:
	def __init__(self, path):
		self.path = path
		self.teams = {}
		self.ground = {}
		self.data = None

		self.parse()

	def parse(self):
		self.data = pd.read_csv(self.path)

		for i, row in self.data.iterrows():

			if row['home_team'] not in self.teams:
				self.teams[row['home_team']] = Team(row['home_team'])
			if row['away_team'] not in self.teams:
				self.teams[row['away_team']] = Team(row['away_team'])

			self.result(row['home_team'], row['away_team'], 
				row['home_score'], row['away_score'])

		for t in self.teams:
			team = self.teams[t]
			print("{} are {}-{}-{}".format(team.name, team.wins, team.losses, team.ties))

	def result(self, home, away, s1, s2):
		if s1 > s2:
			self.teams[home].wins += 1
			self.teams[away].losses += 1
		elif s1 == s2:
			self.teams[home].ties += 1
			self.teams[away].ties += 1
		else:
			self.teams[home].losses += 1
			self.teams[away].wins += 1

p = Parser('games_data/regular_season/reg_games_2018.csv')