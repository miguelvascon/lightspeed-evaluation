"""Shared golden fakes. Judge and lightspeed-stack servers live in later WPs."""

from tests.golden.harness._http_server import Router, prefix, serve

__all__ = ["Router", "prefix", "serve"]
