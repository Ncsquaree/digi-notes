import os
from modules.utils.task_manager import TaskManager, TaskStatus


def test_task_manager_in_memory(monkeypatch):
    # ensure redis not used
    import modules.utils.task_manager as tm_mod
    monkeypatch.setattr(tm_mod, 'redis', None)
    TaskManager._instance = None
    tm = TaskManager.get_instance()
    tid = 'task1'
    obj = tm.create_task(tid, 'u1', 'n1')
    assert obj['task_id'] == tid
    got = tm.get_task(tid)
    assert got is not None

    updated = tm.update_status(tid, TaskStatus.PROCESSING, progress_pct=10, current_step='start')
    assert updated['status'] == TaskStatus.PROCESSING.value

    prog = tm.update_progress(tid, 'step1', 20, {'ok': True})
    assert prog['progress_pct'] == 20

    tm.mark_step_complete(tid, 'step1', {'r': 1})
    t = tm.get_task(tid)
    assert 'step1' in t.get('steps_completed', [])

    tm.mark_step_failed(tid, 'step2', 'error')
    t2 = tm.get_task(tid)
    assert any('step2' in str(e.get('step') if isinstance(e, dict) else e) for e in t2.get('steps_failed', []))

    tm.complete_task(tid, {'final': True})
    t3 = tm.get_task(tid)
    assert t3['status'] == TaskStatus.COMPLETED.value

    # fail non-existent
    assert tm.update_status('nope', TaskStatus.COMPLETED) is None
