"""Stage 1 tests — extract_paths with a stub PyMuPDF document."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.fixtures import default_config


@contextmanager
def _stub_pdf(monkeypatch, n_pages=1, drawings=None, text_blocks=None, page_size=(600, 400)):
    """Patch fitz.open with a minimal stub that mimics the surface we use."""
    drawings = drawings if drawings is not None else []
    text_blocks = text_blocks if text_blocks is not None else []

    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Rect:
        def __init__(self, x0, y0, x1, y1):
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
            self.width = x1 - x0
            self.height = y1 - y0

    class Page:
        def __init__(self):
            self.rect = Rect(0, 0, page_size[0], page_size[1])

        def get_drawings(self):
            return drawings

        def get_text(self, mode):
            assert mode == "dict"
            return {"blocks": [
                {"type": 0, "lines": [{"spans": [{"text": tb["text"], "bbox": tb["bbox"]}]}]}
                for tb in text_blocks
            ]}

    class Doc:
        def __init__(self):
            self._pages = [Page() for _ in range(n_pages)]
            self.page_count = n_pages

        def __getitem__(self, i):
            return self._pages[i]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(_path):
        return Doc()

    import stages.extract_paths as ep_mod

    monkeypatch.setattr(ep_mod.fitz, "open", fake_open)
    # Also patch the Pt class used when constructing drawings via our builder
    yield Pt, Rect


def _line_drawing(Pt, p1, p2, width=1.0, color=(0, 0, 0), dashes="[] 0"):
    return {
        "items": [("l", Pt(*p1), Pt(*p2))],
        "color": color,
        "fill": None,
        "width": width,
        "dashes": dashes,
    }


def test_extract_raises_for_missing_pdf(tmp_path):
    from stages.extract_paths import extract_paths

    with pytest.raises(FileNotFoundError):
        extract_paths(tmp_path / "nope.pdf", None, default_config())


def test_extract_single_page_lines_flips_y(monkeypatch, tmp_path):
    from stages.extract_paths import extract_paths

    config = default_config()
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")  # placeholder; fitz is stubbed

    with _stub_pdf(monkeypatch, n_pages=1, page_size=(600, 400)) as (Pt, _Rect):
        drawings = [_line_drawing(Pt, (10, 50), (110, 50), width=1.0)]

        def reopen(_path):
            class Doc:
                page_count = 1

                def __enter__(self): return self

                def __exit__(self, *a): return False

                def __getitem__(self, i):
                    return _page_with(drawings, [], Pt)

            return Doc()

        import stages.extract_paths as ep_mod
        monkeypatch.setattr(ep_mod.fitz, "open", reopen)

        result = extract_paths(pdf_path, None, config)
        assert result["page_size"] == [600, 400]
        assert len(result["paths"]) == 1
        pts = result["paths"][0]["points"]
        assert pts[0] == [10.0, 350.0]
        assert pts[1] == [110.0, 350.0]
        assert result["paths"][0]["is_dashed"] is False


def _page_with(drawings, text_blocks, Pt):
    class Rect:
        def __init__(self, x0, y0, x1, y1):
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
            self.width = x1 - x0
            self.height = y1 - y0

    class Page:
        rect = Rect(0, 0, 600, 400)

        def get_drawings(self_inner):
            return drawings

        def get_text(self_inner, mode):
            return {"blocks": [
                {"type": 0, "lines": [{"spans": [{"text": tb["text"], "bbox": tb["bbox"]}]}]}
                for tb in text_blocks
            ]}

    return Page()


def test_extract_multi_page_without_selection_raises(monkeypatch, tmp_path):
    from stages.extract_paths import extract_paths
    import stages.extract_paths as ep_mod

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def reopen(_path):
        class Doc:
            page_count = 3

            def __enter__(self): return self

            def __exit__(self, *a): return False

            def __getitem__(self, i):
                return _page_with([_line_drawing(Pt, (0, 0), (1, 1))], [], Pt)

        return Doc()

    monkeypatch.setattr(ep_mod.fitz, "open", reopen)
    with pytest.raises(ValueError, match="Multi-page"):
        extract_paths(pdf_path, None, default_config())


def test_extract_raises_on_empty_page(monkeypatch, tmp_path):
    from stages.extract_paths import extract_paths
    import stages.extract_paths as ep_mod

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def reopen(_path):
        class Doc:
            page_count = 1

            def __enter__(self): return self

            def __exit__(self, *a): return False

            def __getitem__(self, i):
                return _page_with([], [], Pt)

        return Doc()

    monkeypatch.setattr(ep_mod.fitz, "open", reopen)
    with pytest.raises(ValueError, match="zero vector drawings"):
        extract_paths(pdf_path, None, default_config())


def test_extract_text_blocks_flipped(monkeypatch, tmp_path):
    from stages.extract_paths import extract_paths
    import stages.extract_paths as ep_mod

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    tbs = [{"text": "BATHROOM", "bbox": (100, 100, 200, 120)}]

    def reopen(_path):
        class Doc:
            page_count = 1

            def __enter__(self): return self

            def __exit__(self, *a): return False

            def __getitem__(self, i):
                return _page_with([_line_drawing(Pt, (0, 0), (10, 0))], tbs, Pt)

        return Doc()

    monkeypatch.setattr(ep_mod.fitz, "open", reopen)
    result = extract_paths(pdf_path, None, default_config())
    assert len(result["text_blocks"]) == 1
    tb = result["text_blocks"][0]
    assert tb["text"] == "BATHROOM"
    # Y flipped: original top-left (100,100,200,120) with page height 400 → (100, 280, 200, 300)
    assert tb["bbox"] == [100.0, 280.0, 200.0, 300.0]
