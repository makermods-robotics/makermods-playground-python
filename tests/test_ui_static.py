import re
from html.parser import HTMLParser

from fastapi.routing import APIRoute

from makermods_playground.ui.app import STATIC, create_app


class Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        self.refs += [v for k, v in attrs if k in ("src", "href") and v and v.startswith("/static/")]


def static_path(url: str):
    return STATIC / url.split("?")[0].removeprefix("/static/")


def test_every_static_asset_exists():
    parser = Refs()
    parser.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    assert len(parser.refs) > 10
    refs = list(parser.refs)
    for js in STATIC.glob("*.js"):
        text = js.read_text(encoding="utf-8")
        refs += [f"/static/{m}" for m in re.findall(r'from "\./([^"]+)"', text)]
        refs += re.findall(r'"(/static/[^"]+)"', text)
    for css in STATIC.rglob("*.css"):
        refs += re.findall(r'url\("(/static/[^"]+)"\)', css.read_text(encoding="utf-8"))
    missing = sorted({r for r in refs if not static_path(r).is_file()})
    assert not missing


def test_every_api_path_the_page_calls_is_a_route():
    app = create_app()
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    called = set()
    for js in STATIC.glob("*.js"):
        for path in re.findall(r'["`](/api/[^"`?]+)', js.read_text(encoding="utf-8")):
            called.add(re.sub(r"\$\{[^}]*\}", "x", path))
    assert {"/api/session", "/api/status", "/api/run", "/api/stop", "/api/heartbeat", "/api/release",
            "/api/robots/x/select", "/api/identify", "/api/recordings/x/x", "/api/devices/cameras/x/stream"} <= called
    unknown = sorted(p for p in called if not any(r.path_regex.match(p) for r in routes))
    assert not unknown


def test_the_guide_does_not_call_a_robot_the_first_when_robots_exist():
    robots = (STATIC / "robots.js").read_text(encoding="utf-8")
    tutorial = (STATIC / "tutorial.js").read_text(encoding="utf-8")
    assert "count: robots.length" in robots                       # "pg:robot" carries how many robots exist
    assert re.search(r'addEventListener\("pg:robot".*detail\?\.count', tutorial)
    assert re.search(r"current === 0 && hasRobots \? TITLE_WITH_ROBOTS", tutorial)
    assert '"Add your first robot"' in tutorial and 'TITLE_WITH_ROBOTS = "Add another robot"' in tutorial
