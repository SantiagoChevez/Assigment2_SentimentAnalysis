import math
#import vectorized dataset

# Resolve base directories independent of current working directory
from xml.parsers.expat import model


BASE_DIR = Path(__file__).resolve().parent               # .../phase_2
REPO_ROOT = BASE_DIR.parent                              # .../Assigment2_SentimentAnalysis
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = REPO_ROOT / "datasets"

def model_predict(df):
    # load model weights
    model.load_weights(MODELS_DIR / ".pth") #TODO: specify model file

    # model inference

    # return output (series or dataframe)

def buy_rule(stock):
    # calculate number of shares to buy based on available balance and stock price
    alpha = stock['pred_impact_score']
    s = stock['pred_impact_score']

    balance = stock['balance']
    price = stock['price']

    shares = max(1, math.floor((alpha * s / 100) * balance / price))
    return shares

def sell_rule(stock):
    #TODO: implement sell logic


def trading_rules(stock):
    if stock['pred_impact_score'] > 0 and stock[balance]/stock['price'] > 0:
        return buy_rule()
    elif stock['pred_impact_score'] < 0 and stock['owned_shares'] > 0:
        return sell_rule()
    else:
        # implement logic in 3.2 so if None is returned, no trade calculations are made
        return None
