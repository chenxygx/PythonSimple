import redis


class Redis(object):
    def __init__(self):
        self.conn = redis.StrictRedis(host="localhost", port=6379, db=0)

    def save_str(self, key, value):
        self.conn.set(key, value)

    def get_str(self, key):
        return self.conn.get(key)
