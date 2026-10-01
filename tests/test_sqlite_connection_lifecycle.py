"""Regression coverage for transactions whose SQLite connections must close."""

from contextlib import closing
import sqlite3

import pytest

from chemical_properties import ChemicalDatabase, SmilesResolution
from property_resolution.runtime_cache import SQLiteJSONCache
from property_resolution.runtime_locks import SQLiteLeaseLock


@pytest.fixture
def connections(monkeypatch):
    opened = []
    connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            return super().close()

    def tracked_connect(*args, **kwargs):
        kwargs['factory'] = TrackedConnection
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, 'connect', tracked_connect)
    yield opened
    assert opened
    assert all(connection.closed for connection in opened)


def test_json_cache_closes_reads_writes_and_rolls_back_failures(tmp_path, connections):
    path = tmp_path/'cache.sqlite'
    cache = SQLiteJSONCache(path, 'fixture')
    cache.set('one', {'value':1})
    assert cache.get('one') == {'value':1}
    assert cache.get('missing') is None
    assert cache.count() == 1
    assert list(cache.items()) == [('one', {'value':1})]
    iterator = cache.items()
    next(iterator)
    iterator.close()
    with pytest.raises(RuntimeError, match='rollback'):
        with cache._connect() as connection:
            connection.execute("DELETE FROM runtime_json_cache")
            raise RuntimeError('rollback')
    assert cache.get('one') == {'value':1}
    cache.delete('one')
    assert cache.count() == 0


def test_lease_connections_close_after_acquire_renew_release_and_sql_error(tmp_path, connections):
    lock = SQLiteLeaseLock(tmp_path/'locks.sqlite', 'fixture', 'resource')
    with lock:
        assert lock._renew()
        assert not lock._try_acquire()
    with closing(sqlite3.connect(lock.path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM runtime_locks').fetchone()[0] == 0
        connection.execute('DROP TABLE runtime_locks')
        connection.commit()
    with pytest.raises(sqlite3.OperationalError):
        lock._try_acquire()


@pytest.mark.parametrize('kind', ['cache', 'lock'])
def test_connection_closes_when_pragma_setup_fails(tmp_path, connections, monkeypatch, kind):
    owner = (SQLiteJSONCache(tmp_path/'cache.sqlite', 'fixture') if kind == 'cache'
             else SQLiteLeaseLock(tmp_path/'locks.sqlite', 'fixture', 'resource'))
    connection_type = type(connections[0])
    execute = connection_type.execute

    def fail_setup(self, sql, *args, **kwargs):
        if sql.startswith('PRAGMA busy_timeout'):
            raise sqlite3.OperationalError('setup failed')
        return execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(connection_type, 'execute', fail_setup)
    with pytest.raises(sqlite3.OperationalError, match='setup failed'):
        with owner._connect():
            pytest.fail('Connection setup unexpectedly succeeded')


def test_smiles_cache_closes_schema_hits_misses_and_negative_cache(tmp_path, connections, monkeypatch):
    database = ChemicalDatabase(enable_online=False)
    database._smiles_cache_path = tmp_path/'smiles.sqlite'
    monkeypatch.setattr(database, '_opsin_runtime_version', lambda:'fixture-version')
    assert database._cached_smiles('missing') is None
    database._cache_smiles(SmilesResolution(
        smiles='CCO', source='chemicals', method='chemicals_metadata',
        quality=.99, notes='fixture', identifier='ethanol',
    ), ['ethanol'])
    assert database._cached_smiles('ethanol').smiles == 'CCO'
    database._record_opsin_negative('unknown', persist=True)
    database._opsin_failure_memo.clear()
    assert database._opsin_negative_cached('unknown')
    assert not database._opsin_negative_cached('other')
