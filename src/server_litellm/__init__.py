"""
LiteLLM MCP Server package initialization.
"""
from . import server
import asyncio

def main():
	"""Main entry point for the package."""
	# Run the async main function properly with asyncio
	return asyncio.run(server.main())

# Expose important items at package level
__all__ = ['main', 'server']