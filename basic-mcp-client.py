#!/usr/bin/env python3
“””
Basic MCP Client for Streamable HTTP Protocol
This example demonstrates connecting to an MCP server using Streamable HTTP transport.
“””

import asyncio
import json
from typing import Any, Dict

# Import MCP client components

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

class BasicMCPClient:
def **init**(self, server_url: str = “http://localhost:8000/mcp”):
self.server_url = server_url
self.session = None

```
async def connect(self):
    """Connect to the MCP server using Streamable HTTP."""
    print(f"🔗 Connecting to MCP server at {self.server_url}")
    
    try:
        # Create streamable HTTP client connection
        self.transport = streamablehttp_client(self.server_url)
        read_stream, write_stream, _ = await self.transport.__aenter__()
        
        # Create session with the streams
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        
        # Initialize the connection
        init_result = await self.session.initialize()
        print("✅ Connected successfully!")
        print(f"📋 Server: {init_result.serverInfo.name} v{init_result.serverInfo.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def disconnect(self):
    """Disconnect from the MCP server."""
    if self.session:
        await self.session.__aexit__(None, None, None)
    if hasattr(self, 'transport'):
        await self.transport.__aexit__(None, None, None)
    print("🔌 Disconnected from server")

async def list_tools(self):
    """List all available tools from the server."""
    if not self.session:
        print("❌ Not connected to server")
        return
    
    try:
        tools_response = await self.session.list_tools()
        print("\n🛠️ Available Tools:")
        for tool in tools_response.tools:
            print(f"  📌 {tool.name}: {tool.description}")
        return tools_response.tools
    except Exception as e:
        print(f"❌ Error listing tools: {e}")

async def list_resources(self):
    """List all available resources from the server."""
    if not self.session:
        print("❌ Not connected to server")
        return
    
    try:
        resources_response = await self.session.list_resources()
        print("\n📊 Available Resources:")
        for resource in resources_response.resources:
            print(f"  📁 {resource.uri}: {resource.name}")
        return resources_response.resources
    except Exception as e:
        print(f"❌ Error listing resources: {e}")

async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
    """Call a specific tool with given arguments."""
    if not self.session:
        print("❌ Not connected to server")
        return
    
    try:
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"📥 Arguments: {arguments}")
        
        result = await self.session.call_tool(tool_name, arguments)
        print(f"📤 Result: {json.dumps(result.content, indent=2)}")
        return result
    except Exception as e:
        print(f"❌ Error calling tool {tool_name}: {e}")

async def read_resource(self, resource_uri: str):
    """Read a specific resource from the server."""
    if not self.session:
        print("❌ Not connected to server")
        return
    
    try:
        print(f"\n📖 Reading resource: {resource_uri}")
        
        result = await self.session.read_resource(resource_uri)
        print(f"📄 Content:")
        for content in result.contents:
            if hasattr(content, 'text'):
                print(content.text)
            else:
                print(content)
        return result
    except Exception as e:
        print(f"❌ Error reading resource {resource_uri}: {e}")
```

async def demo_client():
“”“Demonstrate basic MCP client functionality.”””
client = BasicMCPClient()

```
# Connect to server
if not await client.connect():
    return

try:
    # List available tools and resources
    await client.list_tools()
    await client.list_resources()
    
    print("\n" + "="*50)
    print("🧪 Running Demo Operations")
    print("="*50)
    
    # Demo 1: Add numbers
    await client.call_tool("add_numbers", {"a": 15, "b": 27})
    
    # Demo 2: Get server status
    await client.call_tool("get_server_status", {})
    
    # Demo 3: Send a notification
    await client.call_tool("send_notification", {
        "message": "Hello from MCP client!",
        "priority": "high",
        "user_id": 1
    })
    
    # Demo 4: Create a new user
    await client.call_tool("create_user", {
        "name": "Charlie",
        "email": "charlie@example.com"
    })
    
    # Demo 5: Read resources
    await client.read_resource("users://all")
    await client.read_resource("user://1")
    await client.read_resource("notifications://recent")
    
    print("\n✅ Demo completed successfully!")
    
except Exception as e:
    print(f"❌ Demo error: {e}")

finally:
    # Always disconnect
    await client.disconnect()
```

# === Interactive Mode ===

async def interactive_mode():
“”“Run an interactive session with the MCP server.”””
client = BasicMCPClient()

```
if not await client.connect():
    return

try:
    tools = await client.list_tools()
    resources = await client.list_resources()
    
    print("\n" + "="*50)
    print("🎮 Interactive MCP Client Mode")
    print("="*50)
    print("Commands:")
    print("  'tools' - List available tools")
    print("  'resources' - List available resources")
    print("  'call <tool_name>' - Call a tool (you'll be prompted for arguments)")
    print("  'read <resource_uri>' - Read a resource")
    print("  'help' - Show this help")
    print("  'quit' - Exit")
    print()
    
    while True:
        try:
            command = input("MCP> ").strip()
            
            if command == "quit":
                break
            elif command == "help":
                print("Available commands: tools, resources, call <tool_name>, read <resource_uri>, help, quit")
            elif command == "tools":
                await client.list_tools()
            elif command == "resources":
                await client.list_resources()
            elif command.startswith("call "):
                tool_name = command[5:].strip()
                if tool_name:
                    # Get tool arguments from user
                    print(f"Calling tool: {tool_name}")
                    args_input = input("Enter arguments as JSON (or press Enter for empty): ").strip()
                    try:
                        args = json.loads(args_input) if args_input else {}
                        await client.call_tool(tool_name, args)
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON format")
                else:
                    print("❌ Please specify a tool name")
            elif command.startswith("read "):
                resource_uri = command[5:].strip()
                if resource_uri:
                    await client.read_resource(resource_uri)
                else:
                    print("❌ Please specify a resource URI")
            elif command:
                print(f"❌ Unknown command: {command}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            break
            
finally:
    await client.disconnect()
```

# === MAIN EXECUTION ===

if **name** == “**main**”:
print(“🚀 MCP Client for Streamable HTTP”)
print(“Make sure the MCP server is running on localhost:8000”)
print()

```
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    # Run interactive mode
    asyncio.run(interactive_mode())
else:
    # Run demo mode
    print("Running demo mode. Use --interactive for interactive mode.")
    asyncio.run(demo_client())
```
