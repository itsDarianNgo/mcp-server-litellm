# src/server_litellm/server.py
import os
import logging
from typing import List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from litellm import completion
from dotenv import load_dotenv

# Load environment variables from .env file or system
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("litellm-server")

# Initialize the server
app = Server("litellm-server")

# Ensure the OPENAI_API_KEY is set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
	raise EnvironmentError("Missing OpenAI API Key. Ensure it is set in the .env file or environment variables.")

@app.list_tools()
async def list_tools() -> List[Tool]:
	"""
	List available tools for LiteLLM server.
	"""
	return [
		Tool(
			name="complete",
			description="Send a completion request to a specified LLM model.",
			inputSchema={
				"type": "object",
				"properties": {
					"model": {
						"type": "string",
						"description": "The LLM model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')."
					},
					"messages": {
						"type": "array",
						"description": "An array of conversation messages, each with 'role' and 'content'.",
						"items": {
							"type": "object",
							"properties": {
								"role": {"type": "string", "description": "The role in the conversation (e.g., 'user', 'assistant')."},
								"content": {"type": "string", "description": "The content of the message."}
							},
							"required": ["role", "content"]
						}
					}
				},
				"required": ["model", "messages"]
			}
		)
	]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
	"""
	Handle the 'complete' tool calls.
	"""
	if name != "complete":
		raise ValueError(f"Unknown tool: {name}")

	try:
		# Extract and validate arguments
		model = arguments.get("model")
		messages = arguments.get("messages", [])

		if not isinstance(messages, list):
			raise ValueError("The 'messages' argument must be a list of objects with 'role' and 'content' fields.")

		# Ensure all messages have 'role' and 'content'
		for message in messages:
			if not isinstance(message, dict) or "role" not in message or "content" not in message:
				raise ValueError(f"Each message must have 'role' and 'content'. Invalid message: {message}")

		# Log the input arguments for debugging
		logger.debug(f"Model: {model}, Messages: {messages}")

		# Call LiteLLM's completion function
		response = completion(
			model=model,
			messages=messages
		)

		# Extract the response text
		text = response.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response content.")

		# Return the response in MCP format
		return [TextContent(type="text", text=text)]

	except Exception as e:
		logger.error(f"Error during LiteLLM completion: {e}")
		raise RuntimeError(f"LLM API error: {e}")

async def run_server():
	"""
	Run the LiteLLM MCP server.
	"""
	async with stdio_server() as (read_stream, write_stream):
		await app.run(read_stream, write_stream, app.create_initialization_options())

def main():
	"""
	Main entry point for the server.
	"""
	import asyncio
	asyncio.run(run_server())
