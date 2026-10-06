from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['api'] == 'running'


def test_scan_post(capsys):
    response = client.post('/scan-post', json={'text': 'My name is Kavya and email is kavya@gmail.com'})
    assert response.status_code == 200
    payload = response.json()
    assert 'risk_score' in payload
    assert 'risk_level' in payload
    assert payload['risk_level'] in {'LOW', 'MEDIUM', 'HIGH'}
    assert 'kavya@gmail.com' not in capsys.readouterr().out


def test_invalid_request_handling():
    response = client.post('/scan-post', json={})
    assert response.status_code == 422


def test_profile_returns_service_unavailable_without_database(monkeypatch):
    from backend import main as backend_main

    def unavailable_database(username):
        raise RuntimeError('Database is unavailable')

    monkeypatch.setattr(backend_main, 'get_user', unavailable_database)

    response = client.get('/profile/mahimashree')

    assert response.status_code == 503
    assert response.json()['detail'] == 'Database is unavailable'


def test_save_post_without_real_db(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(backend_main, 'get_user', lambda username: None)
    monkeypatch.setattr(backend_main, 'create_user', lambda username, bio='': 42)
    monkeypatch.setattr(backend_main, 'save_post_to_db', lambda user_id, content, risk_level, risk_score: None)

    response = client.post(
        '/save-post',
        json={
            'username': 'alice',
            'content': 'hello world',
            'risk_level': 'LOW',
            'risk_score': 10,
        },
    )
    assert response.status_code == 200
    assert response.json()['message'] == 'Post saved successfully'
