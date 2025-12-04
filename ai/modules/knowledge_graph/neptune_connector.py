import os
import threading
import time
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, Retrying

try:
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.structure.graph import Graph
    from gremlin_python.driver.protocol import GremlinServerError
    from gremlin_python.driver.client import Client
except Exception:
    # Import errors will be surfaced when the connector is used at runtime
    DriverRemoteConnection = None
    Graph = None
    GremlinServerError = Exception
    Client = None

import boto3

from modules.utils import get_logger

LOG = get_logger()


class NeptuneConfigError(Exception):
    pass


class NeptuneConnectionError(Exception):
    pass


class NeptuneQueryError(Exception):
    pass


class NeptuneConnector:
    """Singleton wrapper for AWS Neptune Gremlin client.

    Lazy-initializes a DriverRemoteConnection and Graph traversal source (`g`).
    Supports basic IAM/static auth selection and provides health checks and
    graceful close. Retries on transient errors using tenacity.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        # Validate config
        self.endpoint = os.getenv('NEPTUNE_ENDPOINT')
        port = os.getenv('NEPTUNE_PORT')
        if not self.endpoint or not port:
            raise NeptuneConfigError('NEPTUNE_ENDPOINT and NEPTUNE_PORT must be set')
        try:
            self.port = int(port)
            if not (1 <= self.port <= 65535):
                raise ValueError()
        except Exception:
            raise NeptuneConfigError('NEPTUNE_PORT must be an integer 1-65535')

        self.use_iam = os.getenv('NEPTUNE_USE_IAM', 'false').lower() in ('1', 'true', 'yes')
        self.username = os.getenv('NEPTUNE_USERNAME')
        self.password = os.getenv('NEPTUNE_PASSWORD')
        self.pool_size = int(os.getenv('NEPTUNE_POOL_SIZE', '10'))

        # optional gremlin client for pooling (best-effort)
        self._client = None
        if Client is not None:
            try:
                # try common arg name, fall back if driver expects different parameter
                try:
                    self._client = Client(self._build_url(), 'g', max_workers=self.pool_size)
                except TypeError:
                    try:
                        self._client = Client(self._build_url(), 'g', pool_size=self.pool_size)
                    except Exception:
                        self._client = None
                if self._client:
                    LOG.info('neptune_client_initialized', extra={'pool_size': self.pool_size})
            except Exception:
                LOG.warning('neptune_client_init_failed', extra={'pool_size': self.pool_size})

        self._connection = None
        self._g = None
        self._graph = None

        self._connected = False

        # retry config
        attempts = int(os.getenv('NEPTUNE_RETRY_ATTEMPTS', '3'))
        multiplier = int(os.getenv('NEPTUNE_RETRY_MULTIPLIER', '2'))
        max_wait = int(os.getenv('NEPTUNE_RETRY_MAX_WAIT', '10'))
        self._retry_kwargs = dict(stop=stop_after_attempt(attempts), wait=wait_exponential(multiplier=multiplier, max=max_wait))

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = NeptuneConnector()
                    try:
                        cls._instance._connect_with_retry()
                    except Exception as e:
                        LOG.warning('neptune_connect_failed_on_init', extra={'error': str(e)})
        return cls._instance

    def _build_url(self):
        scheme = 'wss'
        return f"{scheme}://{self.endpoint}:{self.port}/gremlin"

    def _connect_with_retry(self):
        url = self._build_url()
        LOG.info('neptune_connect_attempt', extra={'endpoint': self.endpoint, 'port': self.port, 'use_iam': self.use_iam, 'url': url})
        # build a retryer based on instance config
        retryer = Retrying(stop=self._retry_kwargs.get('stop'), wait=self._retry_kwargs.get('wait'), retry=retry_if_exception_type(NeptuneConnectionError), reraise=True)
        for attempt in retryer:
            with attempt:
                try:
                    # DriverRemoteConnection is the standard way to connect
                    if DriverRemoteConnection is None:
                        raise NeptuneConnectionError('gremlin_python not available')

                    if self.use_iam:
                        # Attempt to use AWS SigV4 via boto3 credentials. This requires
                        # the environment to provide AWS credentials and a signing helper
                        # (not part of gremlin_python). Many environments include
                        # neptune-python-utils which provides a SigV4Auth wrapper; try to use it.
                        try:
                            from neptune_python_utils.gremlin_driver import signed_driver_remote_connection
                            # signed_driver_remote_connection builds a DriverRemoteConnection that signs requests
                            self._connection = signed_driver_remote_connection(self._build_url())
                        except Exception:
                            # Fallback: attempt plain connection and hope for network-level auth
                            LOG.warning('neptune_iam_not_supported', extra={'note': 'neptune_python_utils not installed; attempting plain connection'})
                            self._connection = DriverRemoteConnection(self._build_url(), 'g')
                    else:
                        # static auth: pass username/password if provided (DriverRemoteConnection supports username/password args)
                        if self.username and self.password:
                            try:
                                self._connection = DriverRemoteConnection(self._build_url(), 'g', username=self.username, password=self.password)
                            except TypeError:
                                # Older gremlin_python may not accept username/password; fallback
                                self._connection = DriverRemoteConnection(self._build_url(), 'g')
                        else:
                            self._connection = DriverRemoteConnection(self._build_url(), 'g')

                    self._graph = Graph()
                    self._g = self._graph.traversal().withRemote(self._connection)
                    self._connected = True
                    LOG.info('neptune_connected', extra={'endpoint': self.endpoint, 'port': self.port, 'use_iam': self.use_iam})
                    return
                except Exception as exc:
                    LOG.exception('neptune_connect_error', exc_info=True)
                    raise NeptuneConnectionError(str(exc))

    def get_traversal(self):
        if not self._connected or self._g is None:
            # attempt reconnect
            try:
                self._connect_with_retry()
            except Exception as e:
                raise NeptuneConnectionError(f'Failed to connect to Neptune: {e}')
        return self._g

    def execute(self, traversal_callable):
        """Execute a callable that accepts a traversal `g` and performs operations.

        Execution is retried according to instance retry settings.
        """
        retryer = Retrying(stop=self._retry_kwargs.get('stop'), wait=self._retry_kwargs.get('wait'), retry=retry_if_exception_type(NeptuneQueryError), reraise=True)
        for attempt in retryer:
            with attempt:
                g = self.get_traversal()
                try:
                    return traversal_callable(g)
                except GremlinServerError as e:
                    LOG.exception('neptune_query_server_error', exc_info=True)
                    raise NeptuneQueryError(str(e))
                except Exception as e:
                    LOG.exception('neptune_query_error', exc_info=True)
                    raise NeptuneQueryError(str(e))

    def health_check(self, timeout: int = None) -> bool:
        timeout = timeout or int(os.getenv('NEPTUNE_HEALTH_CHECK_TIMEOUT', '5'))
        try:
            g = self.get_traversal()
            # lightweight check: count vertices up to 1
            start = time.time()
            # use a short call that should return quickly
            count = g.V().limit(1).count().next()
            elapsed = time.time() - start
            LOG.info('neptune_health_check', extra={'endpoint': self.endpoint, 'count_sample': int(count), 'elapsed_ms': int(elapsed * 1000)})
            return True
        except Exception as e:
            LOG.exception('neptune_health_failed', exc_info=True)
            return False

    def close(self):
        try:
            if self._connection:
                try:
                    self._connection.close()
                except Exception:
                    pass
            self._connected = False
            LOG.info('neptune_connection_closed', extra={'endpoint': self.endpoint})
        except Exception:
            LOG.exception('neptune_close_failed', exc_info=True)
