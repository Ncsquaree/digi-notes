class MockCursor:
    def __init__(self):
        self.queries = []
        self._rows = []

    def execute(self, q, p=None):
        self.queries.append((q, p))

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


class MockConn:
    def __init__(self):
        self.cur = MockCursor()

    def cursor(self):
        return self.cur

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class MockPool:
    def __init__(self):
        self._conn = MockConn()

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass

    def closeall(self):
        pass
