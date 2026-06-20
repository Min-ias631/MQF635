"""main.py helper tests."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import main


class MainHelpersTests(unittest.TestCase):
    def test_print_startup_banner_includes_urls_and_modes(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main._print_startup_banner(
                market_feed="mock",
                execution="sim",
                api_enabled=True,
                api_host="127.0.0.1",
                api_port=8123,
            )

        out = buf.getvalue()
        self.assertIn("market_feed=mock", out)
        self.assertIn("execution=sim", out)
        self.assertIn("Dashboard: http://127.0.0.1:8123/", out)
        self.assertIn("Status JSON: http://127.0.0.1:8123/status", out)

    def test_keep_api_alive_returns_immediately_when_disabled(self) -> None:
        main._keep_api_alive_if_requested(False, api_host="127.0.0.1", api_port=8000)


if __name__ == "__main__":
    unittest.main()
