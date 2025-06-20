#!/usr/bin/env python3
“””
Low-Level MCP Server with Real Server Notifications
This example demonstrates proper server-sent notifications using the low-level MCP SDK.
“””

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# Import low-level MCP components

import mcp.types as types
import mcp.server.stdio
import mcp.server.sse
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

# For Streamable HTTP support

try:
from mcp.server.streamablehttp import StreamableHttpServerTransport
STREAMABLE_HTTP_AVAILABLE = True
except ImportError:
STREAMABLE_HTTP_AVAILABLE = False
print(“⚠️  Streamable HTTP not available, using SSE transport”)

# Global server instance and session tracking

server = Server(“NotificationServer”)
active_sessions = {}  # Track active sessions for sending notifications

# Notification queue for background notifications

notification_queue = asyncio.Queue()

# === SERVER TOOLS ===

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
“”“Handle tool calls”””
print(f”🔧 Tool called: {name} with {arguments}”)

```
if name == "add_numbers":
    a = arguments.get("a", 0)
    b = arguments.get("b", 0)
    result = a + b
    return [types.TextContent(
        type="text", 
        text=f"Result: {result} (calculated {a} + {b})"
    )]

elif name == "start_notification_demo":
    # Start background notifications
    asyncio.create_task(send_background_notifications())
    return [types.TextContent(
        type="text",
        text="✅ Background notification demo started! You should see notifications every 3 seconds."
    )]

elif name == "send_immediate_notification":
    message = arguments.get("message", "Hello from MCP server!")
    priority = arguments.get("priority", "normal")
    
    # Send immediate notification to all active sessions
    await broadcast_notification(message, priority)
    
    return [types.TextContent(
        type="text",
        text=f"📤 Notification sent: '{message}' with priority: {priority}"
    )]

elif name == "long_running_task":
    # Simulate a long task with progress notifications
    total_steps = arguments.get("steps", 5)
    
    for step in range(total_steps):
        # Send progress notification
        await broadcast_progress_notification(
            progress=step + 1,
            total=total_steps,
            message=f"Processing step {step + 1} of {total_steps}"
        )
        
        # Simulate work
        await asyncio.sleep(1)
    
    return [types.TextContent(
        type="text",
        text=f"✅ Long running task completed with {total_steps} steps!"
    )]

else:
    raise ValueError(f"Unknown tool: {name}")
```

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
“”“List available tools”””
return [
types.Tool(
name=“add_numbers”,
description=“Add two numbers together”,
inputSchema={
“type”: “object”,
“properties”: {
“a”: {“type”: “number”, “description”: “First number”},
“b”: {“type”: “number”, “description”: “Second number”}
},
“required”: [“a”, “b”]
}
),
types.Tool(
name=“start_notification_demo”,
description=“Start background notifications every 3 seconds”,
inputSchema={
“type”: “object”,
“properties”: {}
}
),
types.Tool(
name=“send_immediate_notification”,
description=“Send an immediate notification to all connected clients”,
inputSchema={
“type”: “object”,
“properties”: {
“message”: {“type”: “string”, “description”: “Notification message”},
“priority”: {“type”: “string”, “enum”: [“low”, “normal”, “high”], “default”: “normal”}
},
“required”: [“message”]
}
),
types.Tool(
name=“long_running_task”,
description=“Simulate a long-running task with progress notifications”,
inputSchema={
“type”: “object”,
“properties”: {
“steps”: {“type”: “number”, “description”: “Number of steps to process”, “default”: 5}
}
}
)
]

# === SERVER RESOURCES ===

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
“”“List available resources”””
return [
types.Resource(
uri=“server://status”,
name=“Server Status”,
description=“Current server status and active sessions”
),
types.Resource(
uri=“notifications://recent”,
name=“Recent Notifications”,
description=“List of recent notifications sent by the server”
)
]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
“”“Read resource content”””
if uri == “server://status”:
status = {
“server_name”: “NotificationServer”,
“active_sessions”: len(active_sessions),
“current_time”: datetime.now().isoformat(),
“transport”: “streamable_http” if STREAMABLE_HTTP_AVAILABLE else “sse”,
“notifications_enabled”: True
}
return json.dumps(status, indent=2)

```
elif uri == "notifications://recent":
    # This would typically come from a database or log
    recent_notifications = [
        {"id": 1, "message": "Server started", "timestamp": datetime.now().isoformat(), "priority": "normal"},
        {"id": 2, "message": "Background task completed", "timestamp": datetime.now().isoformat(), "priority": "low"},
    ]
    return json.dumps(recent_notifications, indent=2)

else:
    raise ValueError(f"Unknown resource: {uri}")
```

# === NOTIFICATION FUNCTIONS ===

async def broadcast_notification(message: str, priority: str = “normal”):
“”“Send notification to all active sessions”””
print(f”📡 Broadcasting notification: {message} (priority: {priority})”)

```
for session_id, session_data in active_sessions.items():
    try:
        session = session_data["session"]
        # Send a custom notification
        notification = {
            "method": "notifications/message",
            "params": {
                "level": priority,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Use the session to send notification
        await session.send_notification(notification)
        print(f"✅ Notification sent to session {session_id}")
        
    except Exception as e:
        print(f"❌ Failed to send notification to session {session_id}: {e}")
```

async def broadcast_progress_notification(progress: float, total: float, message: str = “”):
“”“Send progress notification to all active sessions”””
print(f”📊 Broadcasting progress: {progress}/{total} - {message}”)

```
for session_id, session_data in active_sessions.items():
    try:
        session = session_data["session"]
        
        # Send progress notification
        await session.send_progress_notification(
            progress_token="background_task",
            progress=progress,
            total=total,
            message=message
        )
        print(f"📈 Progress sent to session {session_id}: {progress}/{total}")
        
    except Exception as e:
        print(f"❌ Failed to send progress to session {session_id}: {e}")
```

async def send_background_notifications():
“”“Send periodic background notifications”””
print(“🔄 Starting background notification task…”)

```
for i in range(10):  # Send 10 notifications
    await asyncio.sleep(3)  # Wait 3 seconds
    
    message = f"Background notification #{i+1} at {datetime.now().strftime('%H:%M:%S')}"
    await broadcast_notification(message, "low")

print("✅ Background notification task completed")
```

# === SESSION MANAGEMENT ===

class SessionManager:
“”“Manage active sessions and their lifecycles”””

```
def __init__(self):
    self.session_counter = 0

async def register_session(self, session):
    """Register a new session"""
    self.session_counter += 1
    session_id = f"session_{self.session_counter}"
    
    active_sessions[session_id] = {
        "session": session,
        "connected_at": datetime.now(),
        "id": session_id
    }
    
    print(f"🟢 Session registered: {session_id}")
    
    # Send welcome notification
    await broadcast_notification(f"New session connected: {session_id}", "normal")
    
    return session_id

async def unregister_session(self, session_id: str):
    """Unregister a session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        print(f"🔴 Session unregistered: {session_id}")
```

session_manager = SessionManager()

# === TRANSPORT SETUP ===

async def run_stdio():
“”“Run server with STDIO transport”””
print(“🚀 Starting MCP server with STDIO transport”)
print(“📡 Server notifications available via session management”)

```
async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
    # Register this session
    session_id = await session_manager.register_session(server)
    
    try:
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="NotificationServer",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )
    finally:
        await session_manager.unregister_session(session_id)
```

async def run_sse():
“”“Run server with SSE transport”””
print(“🚀 Starting MCP server with SSE transport”)
print(“🌐 Server will be available at http://localhost:8000/sse”)
print(“📡 Real server notifications enabled!”)

```
# This would require setting up an HTTP server with SSE endpoints
# For brevity, we'll show the concept
print("⚠️  SSE implementation requires additional HTTP server setup")
print("Use the stdio version or implement with FastAPI + SSE")
```

async def run_streamable_http():
“”“Run server with Streamable HTTP transport”””
if not STREAMABLE_HTTP_AVAILABLE:
print(“❌ Streamable HTTP not available”)
return

```
print("🚀 Starting MCP server with Streamable HTTP transport")
print("🌐 Server will be available at http://localhost:8000/mcp")
print("📡 Real server notifications enabled via SSE upgrade!")

# This is a conceptual implementation
# The actual StreamableHttpServerTransport would handle the HTTP/SSE details
print("⚠️  Full Streamable HTTP implementation requires additional setup")
```

# === MAIN EXECUTION ===

if **name** == “**main**”:
print(”=” * 60)
print(“🔔 MCP Server with Real Server Notifications”)
print(”=” * 60)
print()
print(“Features:”)
print(“📋 Tools: add_numbers, start_notification_demo, send_immediate_notification, long_running_task”)
print(“📊 Resources: server://status, notifications://recent”)
print(“📡 Server Notifications: Real-time notifications and progress updates”)
print()
print(“Transport Options:”)
print(“1. STDIO (default) - Local process communication”)
print(“2. SSE - HTTP Server-Sent Events”)
print(“3. Streamable HTTP - Modern HTTP with SSE upgrade”)
print()

```
import sys
transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"

try:
    if transport == "stdio":
        asyncio.run(run_stdio())
    elif transport == "sse":
        asyncio.run(run_sse())
    elif transport == "streamable-http":
        asyncio.run(run_streamable_http())
    else:
        print(f"❌ Unknown transport: {transport}")
        print("Available: stdio, sse, streamable-http")
        
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
except Exception as e:
    print(f"❌ Server error: {e}")
```
