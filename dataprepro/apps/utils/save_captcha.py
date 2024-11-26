from redis import StrictRedis
from dataprepro.apps.settings.settings_manager import settings


class RedisClient:
    def __init__(self,
                 hostname=settings.get('REDIS_HOST', 'localhost'),
                 port=settings.get('REDIS_PORT', 6379),
                 password=settings.get('REDIS_PASSWORD'),
                 ):
        self.db = StrictRedis(host=hostname, port=port, password=password)

    def add(self, key, value, time=60):
        self.db.set(key, value, time)

    def get(self, key):
        if not self.db.get(key):
            return None
        return self.db.get(key).decode('utf-8')