def test_gradio_app_builds():
    from ui.frontend.app import build_demo

    demo = build_demo()
    assert demo is not None

