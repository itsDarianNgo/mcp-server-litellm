"""
Main entry point for the LiteLLM MCP server.
"""
import asyncio
from server_litellm.server import main

if __name__ == "__main__":
	# Run the async main function with asyncio
	asyncio.run(main())