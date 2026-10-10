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


def test_scan_does_not_treat_emojis_as_sensitive_entities():
    response = client.post(
        '/scan-post',
        json={
            'text': (
                'Hello everyone! 👋 Today I am testing my Smart Microblogging '
                'application. Technology makes communication easier, but '
                'protecting personal information online is important. '
                'Always think before you share! 🔐✨ '
                '#Testing #SmartMicroblogging #OnlinePrivacy #StaySafe'
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['risk_score'] == 0
    assert payload['risk_level'] == 'LOW'
    assert payload['detected_entities']['entities'] == []


def test_scan_does_not_treat_possessive_my_as_a_location():
    response = client.post(
        '/scan-post',
        json={
            'text': (
                'My name is Rahul Sharma. I live at 123 Example Street, '
                'Bengaluru, and my phone number is 9876543210.'
            ),
        },
    )

    assert response.status_code == 200
    locations = response.json()['detected_entities']['locations']
    assert 'Bengaluru' in locations
    assert 'my' not in [location.lower() for location in locations]


def test_scan_does_not_treat_placeholder_as_person():
    response = client.post(
        '/scan-post',
        json={
            'text': 'hii my num is 7855946231 and i am from davanagere and my father name is xyz'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    assert entities['phones'] == ['7855946231']
    assert entities['persons'] == []
    assert 'davanagere' in [location.lower() for location in entities['locations']]


def test_scan_normalizes_duplicate_case_and_rejects_placeholders():
    response = client.post(
        '/scan-post',
        json={
            'text': 'hii my name is Kavana and my father name is xyz and my number is 7854961235'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    assert entities['phones'] == ['7854961235']
    assert entities['persons'] == ['Kavana']
    assert 'xyz' not in [person.lower() for person in entities['persons']]
    assert 'hii' not in [person.lower() for person in entities['persons']]


def test_scan_detects_lowercase_name_from_explicit_name_context():
    response = client.post(
        '/scan-post',
        json={
            'text': 'hi my name is kavana and my friend is manya and my siblings name are deepa keerthana shalini'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    assert 'kavana' in [person.lower() for person in entities['persons']]
    assert 'manya' in [person.lower() for person in entities['persons']]
    assert 'deepa' in [person.lower() for person in entities['persons']]
    assert 'keerthana' in [person.lower() for person in entities['persons']]
    assert 'shalini' in [person.lower() for person in entities['persons']]


def test_scan_detects_names_and_locations_in_user_sentence():
    response = client.post(
        '/scan-post',
        json={
            'text': 'my name is kavana and my mother is xyz and my frd name is manyashree and my brother is abhi and my number is 7845967845 and adhar 8948 7894 7894 and pan is RTDFS7894K and from manglore and from davanagere and from xyz place and usa and canada and india and udupi'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    persons = [person.lower() for person in entities['persons']]
    locations = [location.lower() for location in entities['locations']]

    assert {'kavana', 'manyashree', 'abhi'} <= set(persons)
    assert 'xyz' not in persons
    assert {'manglore', 'davanagere', 'usa', 'canada', 'india', 'udupi'} <= set(locations)
    assert not any(location.startswith('xyz') for location in locations)
    assert entities['phones'] == ['7845967845']
    assert entities['aadhaars'] == ['8948 7894 7894']
    assert entities['pans'] == ['RTDFS7894K']


def test_scan_detects_trailing_comma_and_name_list():
    response = client.post(
        '/scan-post',
        json={
            'text': 'my name is kavana and my mother is xyz and my frd name is manya and my brother is abhi and my number is 7845967845 and adhar 8948 7894 7894 and pan is RTDFS7894K and from manglore and from davanagere and from xyz place and usa and canada and india and udupi ,and keerthana and shalini'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    persons = [person.lower() for person in entities['persons']]
    locations = [location.lower() for location in entities['locations']]

    assert {'kavana', 'manya', 'abhi', 'keerthana', 'shalini'} <= set(persons)
    assert 'xyz' not in persons
    assert {'manglore', 'davanagere', 'usa', 'canada', 'india', 'udupi'} <= set(locations)


def test_scan_treats_tamilnadu_as_location_not_person():
    response = client.post(
        '/scan-post',
        json={
            'text': 'hii my name is kavana and my father name is xyz and my number is 7899090785 and i am from davanagere and i went to xyz and my frd name is manya and my adahr is 7894 7898 7845 and i went to manglore usa india and run to mandya and my college is in bantwal and i am daughter of deepa and my mother native is tamilnadu'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    assert 'tamilnadu' not in [person.lower() for person in entities['persons']]
    assert 'tamilnadu' in [location.lower() for location in entities['locations']]


def test_scan_detects_missing_family_names_and_locations():
    response = client.post(
        '/scan-post',
        json={
            'text': 'i am kavana from davanagere and my father name is xyz and i am a girl and my frd is manya and i am 20 years old and my number is7899030758 and my adhar is 7894 7894 7894 and my frd is from tamilnadu and mhy mother name is deepa and my guide is saritha m and she is from usa and my mother is from goa and my father is from udupi and i have 3 siblings name kavya keerthana shalini'
        },
    )

    assert response.status_code == 200
    entities = response.json()['detected_entities']
    assert 'deepa' in [person.lower() for person in entities['persons']]
    assert 'kavya' in [person.lower() for person in entities['persons']]
    assert 'keerthana' in [person.lower() for person in entities['persons']]
    assert 'shalini' in [person.lower() for person in entities['persons']]
    assert 'goa' in [location.lower() for location in entities['locations']]


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


def test_bookmark_flow(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(
        backend_main,
        'get_user',
        lambda username: {'id': 7, 'username': username, 'bio': '', 'profile_image': ''},
    )
    monkeypatch.setattr(
        backend_main,
        'save_bookmark_to_db',
        lambda user_id, post_id: {'id': 1, 'user_id': user_id, 'post_id': post_id},
    )
    monkeypatch.setattr(
        backend_main,
        'get_bookmarks_for_user',
        lambda username: [{
            'id': 1,
            'post_id': 99,
            'username': username,
            'content': 'Saved for later',
            'timestamp': '2024-01-01T00:00:00',
        }],
    )

    create_response = client.post('/bookmarks', json={'username': 'alice', 'post_id': 99})
    assert create_response.status_code == 200
    assert create_response.json()['bookmarked'] is True

    list_response = client.get('/bookmarks/alice')
    assert list_response.status_code == 200
    assert list_response.json()['bookmarks'][0]['post_id'] == 99


def test_like_post_returns_persisted_state(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(backend_main, 'get_post', lambda post_id: {'id': post_id})
    monkeypatch.setattr(backend_main, 'get_user', lambda username: {'id': 7})
    monkeypatch.setattr(
        backend_main,
        'toggle_post_like',
        lambda user_id, post_id: {'active': True, 'count': 3},
    )

    response = client.post('/post/99/like', json={'username': 'alice'})

    assert response.status_code == 200
    assert response.json() == {'liked': True, 'like_count': 3}


def test_repost_post_returns_persisted_state(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(backend_main, 'get_post', lambda post_id: {'id': post_id})
    monkeypatch.setattr(backend_main, 'get_user', lambda username: {'id': 7})
    monkeypatch.setattr(
        backend_main,
        'toggle_post_repost',
        lambda user_id, post_id: {'active': True, 'count': 2},
    )

    response = client.post('/post/99/repost', json={'username': 'alice'})

    assert response.status_code == 200
    assert response.json() == {'reposted': True, 'repost_count': 2}


def test_send_message_persists_message(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(
        backend_main,
        'get_user',
        lambda username: {'id': 1 if username == 'alice' else 2, 'username': username},
    )
    monkeypatch.setattr(
        backend_main,
        'save_message_to_db',
        lambda sender_id, recipient_id, content: {
            'id': 31,
            'created_at': '2026-10-06T12:00:00',
        },
    )

    response = client.post(
        '/messages',
        json={'username': 'alice', 'recipient': 'bob', 'content': 'Hi there'},
    )

    assert response.status_code == 200
    assert response.json()['id'] == 31
    assert response.json()['sender'] == 'alice'
    assert response.json()['recipient'] == 'bob'
    assert response.json()['content'] == 'Hi there'


def test_message_rejects_empty_content():
    response = client.post(
        '/messages',
        json={'username': 'alice', 'recipient': 'bob', 'content': '   '},
    )

    assert response.status_code == 400
    assert response.json()['detail'] == 'Message cannot be empty.'


def test_read_message_conversations(monkeypatch):
    from backend import main as backend_main

    monkeypatch.setattr(
        backend_main,
        'get_message_conversations',
        lambda username: [{
            'username': 'bob',
            'last_message': 'Hi there',
            'updated_at': '2026-10-06T12:00:00',
        }],
    )

    response = client.get('/messages/alice')

    assert response.status_code == 200
    assert response.json()['conversations'][0]['username'] == 'bob'


def test_profile_rename_updates_existing_user(monkeypatch):
    from backend import main as backend_main

    users = {
        'alice': {'id': 12, 'username': 'alice', 'bio': 'Old bio', 'profile_image': ''},
    }
    rename_calls = []

    def fake_get_user(username):
        return users.get(username)

    def fake_rename_user(current_username, username, bio, profile_image):
        rename_calls.append((current_username, username, bio, profile_image))
        users[username] = {
            'id': users.pop(current_username)['id'],
            'username': username,
            'bio': bio,
            'profile_image': profile_image,
        }

    monkeypatch.setattr(backend_main, 'get_user', fake_get_user)
    monkeypatch.setattr(backend_main, 'rename_user', fake_rename_user)

    response = client.post(
        '/profile',
        json={
            'current_username': 'alice',
            'username': 'alice_new',
            'bio': 'Updated profile',
            'profile_image': 'https://example.com/avatar.png',
        },
    )

    assert response.status_code == 200
    assert response.json()['username'] == 'alice_new'
    assert response.json()['bio'] == 'Updated profile'
    assert 'alice' not in users
    assert rename_calls == [('alice', 'alice_new', 'Updated profile', 'https://example.com/avatar.png')]


def test_profile_rename_rejects_taken_username(monkeypatch):
    from backend import main as backend_main

    users = {
        'alice': {'id': 12, 'username': 'alice', 'bio': '', 'profile_image': ''},
        'bob': {'id': 13, 'username': 'bob', 'bio': '', 'profile_image': ''},
    }

    monkeypatch.setattr(backend_main, 'get_user', lambda username: users.get(username))

    response = client.post(
        '/profile',
        json={'current_username': 'alice', 'username': 'bob', 'bio': '', 'profile_image': ''},
    )

    assert response.status_code == 409
    assert response.json()['detail'] == 'That username is already taken.'


def test_profile_rejects_invalid_username():
    response = client.post(
        '/profile',
        json={'current_username': 'alice', 'username': 'bad username', 'bio': '', 'profile_image': ''},
    )

    assert response.status_code == 400
    assert 'Username must be' in response.json()['detail']
