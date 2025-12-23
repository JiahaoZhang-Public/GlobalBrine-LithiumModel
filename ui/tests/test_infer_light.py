def test_infer_light_not_configured(client):
    r = client.get("/infer_light", params={"lat": 23.05, "lon": -67.25})
    # Dummy mode doesn't implement light inference.
    assert r.status_code in (404, 503)

