"""Review mode and the fits use one label vocabulary."""

from pathlib import Path

from scoring.fit_score import NEGATIVES, PHOTO_NEGATIVES, VOCABULARY

ROOT = Path(__file__).parents[1]


def test_review_mode_and_fit_share_the_vocabulary() -> None:
    classes = VOCABULARY["classes"]
    assert {"boulder", *NEGATIVES, *PHOTO_NEGATIVES, *VOCABULARY["tags"]} <= set(classes)
    assert "boulder" not in PHOTO_NEGATIVES and set(NEGATIVES) <= set(PHOTO_NEGATIVES)
    # Review mode offers the vocabulary's classes (the e2e test checks the panel) and no own list.
    review = (ROOT / "web/src/review.ts").read_text(encoding="utf-8")
    assert 'from "../../scoring/label_vocabulary.json"' in review
