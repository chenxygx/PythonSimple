import requests
import json


class Face():
    def __init__(self, token):
        self.token = token
        self.api_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token=%s"

    def detection(self, path):
        params = "{\"image\":\"http://chenxy.net:81/images/%s\",\"image_type\":\"URL\",\"face_field\":\"age,beauty,expression,faceshape,gender,glasses,landmark,race,quality,facetype\"}" % (
            path)
        request_url = self.api_url % (self.token)
        headers = {'Content-Type': 'application/json'}
        with requests.post(request_url, data=params, headers=headers) as r:
            result = r.json()["result"]["face_list"][0]
            return result
