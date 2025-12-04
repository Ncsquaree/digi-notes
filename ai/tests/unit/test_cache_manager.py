from modules.semantic.cache_manager import CacheManager


def test_cache_get_set(monkeypatch):
    # patch redis.Redis to our mock
    from tests.fixtures.mock_redis import MockRedisClient
    monkeypatch.setattr('redis.Redis', lambda *a, **k: MockRedisClient())
    cm = CacheManager.get_instance()
    parsed = {'a': 1}
    cm.set_summary(parsed, 'brief', {'brief': 'ok'}, ttl=1)
    got = cm.get_summary(parsed, 'brief')
    assert got is not None
    cm.invalidate_summary(parsed, 'brief')
    got2 = cm.get_summary(parsed, 'brief')
    assert got2 is None
