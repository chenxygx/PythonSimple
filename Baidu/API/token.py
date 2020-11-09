import requests
from Tools.redis import Redis

APIKey = "nbMnGlWHynH3YcZTzpxouNUw"
SecretKey = "WqH6MOQR574ZaXFNwnvArIC6zOGwfE5y"
Token_URL = "http://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (
    APIKey, SecretKey)
Sentiment_Token = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token=%s"


class Token(object):
    def __init__(self):
        self.redis = Redis()

    def getToken(self):
        if self.redis.get_str("Token") is None:
            return self.saveToken()
        else:
            return self.redis.get_str("Token")

    def saveToken(self):
        with requests.get(Token_URL) as r:
            result = r.json()["access_token"]
            self.redis.save_str("Token", result)
            return result

    def sentiment(self, text):
        token = self.getToken()
        url = Sentiment_Token % (token)
        param = {"text": text}
        with requests.post(url, json=param) as r:
            result = r.json()
            items = result["items"][0]
            sentiment = {0: "负向", 1: "中性", 2: "正向"}
            return sentiment[items["sentiment"]], items["positive_prob"], items["negative_prob"]
