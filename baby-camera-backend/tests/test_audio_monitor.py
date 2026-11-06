from backend.audio_monitor import CryPolicy, AudioMonitor
import pytest

CRY = {'Baby cry, infant cry': .8}
SCREAM = {'Screaming': .9}


def test_cry_requires_two_windows():
    p = CryPolicy(window_seconds=2)
    assert p.observe(CRY, 2) is None
    assert p.observe(CRY, 4) == 'cry_warning'
    assert p.observe(CRY, 6) is None


def test_scream_requires_one_window():
    assert CryPolicy(window_seconds=2).observe(SCREAM, 2) == 'scream_warning'


def test_gap_breaks_persistence():
    p = CryPolicy(window_seconds=2)
    p.observe(CRY, 2)
    assert p.observe(CRY, 10) is None
    assert p.observe(CRY, 12) == 'cry_warning'


def test_quiet_and_cooldown_both_required():
    p = CryPolicy(window_seconds=2)
    p.observe(SCREAM, 2)
    for at in range(4, 14, 2):
        p.observe({}, at)
    assert not p.latched
    assert p.observe(SCREAM, 14) is None
    assert p.observe(SCREAM, 64) == 'scream_warning'


def test_continuous_cry_does_not_repeat_after_cooldown():
    p = CryPolicy(window_seconds=2)
    p.observe(SCREAM, 2)
    for at in range(4, 130, 2):
        assert p.observe(CRY, at) is None


def test_speech_or_volume_not_used_as_cry():
    p = CryPolicy(window_seconds=2)
    for at in range(2, 20, 2):
        assert p.observe({'Speech': .99, 'Baby cry, infant cry': .1}, at) is None


def test_disconnect_does_not_rearm_episode():
    p = CryPolicy(window_seconds=2)
    p.observe(SCREAM, 2)
    p.reset_gap()
    assert p.observe(SCREAM, 100) is None


def test_missing_model_is_explicit(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('backend.audio_monitor._get_bool', lambda *a: True)
    monitor = AudioMonitor(SimpleNamespace(cameras=[]), None)
    monitor.ffmpeg = 'nonexistent-ffmpeg-file'
    monitor._infer()
    assert monitor.status()['error'] is not None


def test_live_five_second_policy():
    p = CryPolicy()
    assert p.observe(CRY, 5) == 'cry_warning'
    assert p.observe(CRY, 10) is None
    p.observe({}, 15)
    assert p.latched
    p.observe({}, 20)
    assert not p.latched


def test_calibrated_threshold_keeps_cooldown_and_quiet_rearm():
    scores = {'Baby cry, infant cry': .30}
    assert CryPolicy().observe(scores, 5) is None
    p = CryPolicy(cry_threshold=.25)
    assert p.observe(scores, 5) == 'cry_warning'
    assert p.observe(scores, 10) is None
    p.observe({}, 15)
    p.observe({}, 20)
    assert not p.latched
    assert p.observe(scores, 25) is None
    assert p.observe(scores, 65) == 'cry_warning'


@pytest.mark.parametrize('threshold', [0, .249, 1.1, float('nan'), float('inf')])
def test_invalid_calibration_rejected(threshold):
    with pytest.raises(ValueError):
        CryPolicy(cry_threshold=threshold)


def test_monitor_uses_configured_threshold_and_reports_it(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('backend.audio_monitor._get_float', lambda *a: .25)
    monitor = AudioMonitor(SimpleNamespace(cameras=[SimpleNamespace(id='test')]), None)
    assert monitor.status()['cry_threshold'] == .25
    assert monitor.policies['test'].observe({'Baby cry, infant cry': .26}, 5) == 'cry_warning'


def test_bad_config_falls_back_to_default(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('backend.audio_monitor._get_float', lambda *a: float('nan'))
    assert AudioMonitor(SimpleNamespace(cameras=[]), None).status()['cry_threshold'] == .5
