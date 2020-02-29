from tensorflow import keras
import json


def create_model():
	tf_model = keras.models.Sequential()
	tf_model.load_weights("checkpoint_path")
	return tf_model


def get_match():
	match = {"Odds": [0.5, 0.5], "Names": ["", ""]}  # Need to actually get match data soon
	return match


def get_info(players):
	info = players  # Need to get actual player info data sets soon and MUST BE IN PREPROCESSED FORM
	return info


def predict(players):
	info = get_info(players)  # The data set of player info that can be directly inputted into the model
	prediction = model+info
	odds = [prediction.confidence, 1-prediction.confidence]
	return odds
	

def decide_bet(match):
	betting_odds = match["Odds"]  # Will be in the form of dollars won per dollar at stake
	betting_probs = [betting_odds[1]/(betting_odds[0]+betting_odds[1]), betting_odds[0]/(betting_odds[0]+betting_odds[1])]
	players = match["Names"]
	info = get_info(players)
	predicted_odds = predict(info)  # Will be in the form of two decimals that add up to 1
	returns = [betting_odds[0]*predicted_odds[0], betting_odds[1]*predicted_odds[1]]
	chosen = returns.index(max(returns))+1
	amount = (predicted_odds[chosen-1]*(betting_odds[chosen-1]+1)-1)/betting_odds[chosen-1]
	bet = [chosen, amount]
	data_set = [match, betting_odds, betting_probs, players, predicted_odds, returns, chosen, amount]
	return bet, data_set


def place_bet(bet):
	return bet  # Need to actually place bets


def store_data(data_set):
	with open('data.json') as json_file:
		data = json.load(json_file)
		data.append(data_set)
		json.dump(data, json_file)
	return
	
	
def main():
	while True:
		match = get_match()
		bet, data_set = decide_bet(match)
		place_bet(bet)
		store_data(data_set)
	
		
model = create_model()
main()
