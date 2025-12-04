class MockRedisClient:
    def __init__(self):
        self.store = {}
        self.expirations = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v
        if ex:
            self.expirations[k] = ex

    def setex(self, k, ttl, v):
        self.set(k, v, ex=ttl)

    def delete(self, k):
        self.store.pop(k, None)
        self.expirations.pop(k, None)

    def ping(self):
        return True

    def hgetall(self, k):
        return self.store.get(k, {})

    def exists(self, k):
        return 1 if k in self.store else 0

    def flushall(self):
        self.store.clear()
        self.expirations.clear()
