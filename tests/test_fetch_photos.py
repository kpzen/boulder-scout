from typing import Any

from scripts.fetch_photos import choose


def _item(
    iid: str, x0: float, y0: float, when: str, code: str = "EPSG:3006", res: float = 0.4
) -> dict[str, Any]:
    side = 5000.0 if res == 0.4 else 2500.0
    box = [x0, y0, x0 + side, y0 + side]
    props = {"proj:bbox": box, "proj:code": code, "datetime": when, "upplosning": res}
    return {"id": iid, "properties": props}


def test_earliest_in_season_per_square_near_the_tiles() -> None:
    tiles = [(495000.0, 6920000.0, 497500.0, 6922500.0)]
    items = [
        _item("a_mq20", 495000.0, 6920000.0, "2020-06-25T06:54:44Z"),
        _item("a_mq23", 495000.0, 6920000.0, "2023-06-25T10:07:04Z"),  # same day: the newest
        _item("north_mq21", 495000.0, 6922000.0, "2021-05-28T09:00:00Z"),  # earlier in spring
        _item("touching", 490000.0, 6920000.0, "2023-06-25T10:07:04Z"),  # shares an edge only
        _item("north", 495000.0, 6922000.0, "2023-06-25T10:07:04Z"),
        _item("rt90", 495000.0, 6920000.0, "2024-01-01T00:00:00Z", "EPSG:3021"),
        # A town's finer photo inside a_mq23's square: the stage cuts all at one resolution.
        _item("town_mr23", 495000.0, 6920000.0, "2023-06-25T10:07:04Z", res=0.16),
    ]
    assert [it["id"] for it in choose(items, tiles)] == ["a_mq23", "north_mq21"]
