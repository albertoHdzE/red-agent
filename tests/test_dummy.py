from red_agent import say_hi  # Import from package instead of module


def test_say_hi():
    assert say_hi() == "👋 Hello from red-Agent"
