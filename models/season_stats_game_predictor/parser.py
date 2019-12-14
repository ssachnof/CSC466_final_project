import pandas as pd
import numpy as np
import csv
import sys

class Team:
	def __init__(self, name):
		self.wins = 0
		self.home_wins = 0
		self.away_wins = 0
		self.name = name
		self.offense = 0
		self.defense = 0
		self.opoints = 0
		self.dpoints = 0
		self.ranks = {
			'wins': 0,
			'home_wins': 0,
			'away_wins': 0,
			'offense': 0,
			'defense': 0,
			'opoints': 0,
			'dpoints': 0
		}

	def output(self):
		data = self.name + ','
		for r in self.ranks:
			data += '{},'.format(self.ranks[r])
		data += '\n'
		return data


class Parser:
	def __init__(self, year, num_ranks):
		self.path = '../../ryan_data/'
		self.year = year
		self.num_ranks = num_ranks
		self.teams = {}
		self.halfway = {}
		self.parse()

	def parse(self):
		fields = Team('sample').ranks.keys()
		self.parse_path()
		self.parse_games()
		self.parse_pbp()
		for f in fields:
			self.rank(f)

	def parse_path(self):
		self.game_path = self.path + 'game_data/reg_games_{}.csv'.format(self.year)
		self.pbp_path = self.path + 'pbp/reg_pbp_{}.csv'.format(self.year)

	def parse_pbp(self):
		data = pd.io.parsers.read_csv(self.pbp_path).as_matrix()

		for row in data:
			team = row[4]
			defense = row[6]
			play_type = row[25]
			yards = row[26]

			if play_type == 'run' or play_type == 'pass':
				self.teams[team].offense += yards
				self.teams[defense].defense += yards

	def parse_games(self):
		data = pd.read_csv(self.game_path)

		for i, row in data.iterrows():

			if row['home_team'] not in self.teams:
				self.teams[row['home_team']] = Team(row['home_team'])
			if row['away_team'] not in self.teams:
				self.teams[row['away_team']] = Team(row['away_team'])

			if row['home_score'] > row['away_score']:
				self.teams[row['home_team']].wins += 1
				self.teams[row['home_team']].home_wins += 1
			elif row['home_score'] < row['away_score']:
				self.teams[row['away_team']].wins += 1
				self.teams[row['away_team']].away_wins += 1

			self.teams[row['home_team']].opoints += row['home_score']
			self.teams[row['home_team']].dpoints += row['away_score']
			self.teams[row['away_team']].dpoints += row['home_score']
			self.teams[row['away_team']].opoints += row['away_score']

	def rank(self, field):
		low, low_t = 1000000, None
		high, high_t = 0, None

		for team in self.teams:
			t = self.teams[team]
			val = getattr(t, field)

			if low > val:
				low = val
				low_t = team
			if high < val:
				high = val
				high_t = team

		inc = (high-low)/self.num_ranks
		incs = [low+(inc*(1+i)) for i in range(self.num_ranks-1)]
		print(incs)

		for team in self.teams:
			val = getattr(self.teams[team], field)
			r = self.num_ranks-1

			for i in range(self.num_ranks-1):
				if val < incs[i]:
					r = i
					break

			self.teams[team].ranks[field] = r+1
			# setattr(self.teams[team].ranks, field, r)

	def print_teams(self):
		for team in self.teams:
			t = self.teams[team]
			print("""team {}:\n
					wins: {} (Rank {})\n
					home wins: {} (Rank {})\n
					offense: {} (Rank {})\n
					defense: {} (Rank {})\n"""
					.format(t.name, t.wins, t.ranks['wins'], t.home_wins, t.ranks['home_wins'],
						t.offense, t.ranks['offense'], t.defense, t.ranks['defense']))

	def output_classify(self):
		data = pd.read_csv(self.game_path)
		output = open('{}season_data_{}.csv'.format(self.path, self.year), 'w+')
		header = "home,away,rec,op_rec,home_rec,op_away_rec,off,def,op_off,op_def,hop,aop,hdp,adp,div,result\n"
		
		valid = '-1,-1,'
		for _ in range(12):
			valid += '{},'.format(self.num_ranks)
		valid += '2,2\n'


		output.write(header)
		output.write(valid)
		output.write('result\n')
		for i, row in data.iterrows():
			home = self.teams[row['home_team']]
			away = self.teams[row['away_team']]
			div = isDiv(row['home_team'], row['away_team'])
			result = row['home_score'] >= row['away_score']

			o = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
				home.name, away.name, home.ranks['wins'], away.ranks['wins'],
				home.ranks['home_wins'], away.ranks['away_wins'],
				home.ranks['offense'], home.ranks['defense'],
				away.ranks['offense'], away.ranks['defense'], 
				home.ranks['opoints'], away.ranks['opoints'],
				home.ranks['dpoints'], away.ranks['dpoints'], div, result)
			# o = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
			# 	home.name, away.name, home.wins, away.wins,
			# 	home.home_wins, away.away_wins,
			# 	home.offense, home.defense,
			# 	away.offense, away.defense, div, result)
			output.write(o)


	def output_ranks(self):
		output = open('{}rank_data_{}.csv'.format(self.path, self.year), 'w')
		header = "team,rec,home_rec,away_rec,offense,defense,\n"

		output.write(header)
		for t in self.teams:
			o = self.teams[t].output()
			output.write(o)


divs = [{'CLE', 'PIT', 'CIN', 'BAL'}, {'GB', 'MIN', 'DET', 'CHI'},
		{'HOU', 'IND', 'JAX', 'TEN'}, {'NYJ', 'NE', 'MIA', 'BUF'},
		{'OAK', 'KC', 'DEN', 'LAC'}, {'PHI', 'DAL', 'NYG', 'WAS'},
		{'NO', 'CAR', 'TB', 'ATL'}, {'LA', 'SF', 'ARI', 'SEA'}]
def isDiv(home, away):
	for div in divs:
		if home in div:
			if away in div:
				return True
			return False
	return False



# output = open('../../ryan_data/season_data_total.csv', 'a')
# header = "home,away,rec,op_rec,home_rec,op_away_rec,off,def,op_off,op_def,hop,aop,hdp,adp,div,result\n"
		

def main():
	# global output, header
	years = [2014, 2015, 2016, 2017, 2018]
	if len(sys.argv) != 3:
		print("parser.py <year> <num_ranks>\n")
		return 1

	year = sys.argv[1]
	nr = sys.argv[2]
	for y in years:
		print(y)
		p = Parser(y, int(nr))
		# p.print_teams()
		p.output_ranks()
		p.output_classify()

if __name__ == '__main__':
	main()