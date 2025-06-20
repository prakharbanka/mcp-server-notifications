#!/usr/bin/env python3
“””
FastMCP Server with Real Server Notifications
This example shows how to send actual server notifications using FastMCP by accessing the session directly.
Based on the GitHub issue: https://github.com/modelcontextprotocol/python-sdk/issues/953
“””

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any
from pydantic import BaseModel

# Import FastMCP from the official MCP SDK

from mcp.server.fastmcp import FastMCP, Context

# Initialize the MCP server with stateful mode for session management

mcp = FastMCP(“NotificationServer”, stateless_http=False)

# Data storage for demonstration

_server_data = {
“users”: [
{“id”: 1, “name”: “Alice”, “email”: “alice@example.com”},
{“id”: 2, “name”: “Bob”, “email”: “bob@example.com”},
],
“notifications”: [],
“active_tasks”: {}
}

# Track active sessions for broadcasting

_active_sessions = {}

# Pydantic models for complex inputs

class NotificationRequest(BaseModel):
message: str
priority: str = “normal”  # low, normal, high
broadcast: bool = False  # Whether to broadcast to all sessions

class BackgroundTaskRequest(BaseModel):
task_name: str
duration_seconds: int = 10
interval_seconds: int = 2

# === NOTIFICATION HELPER FUNCTIONS ===

async def send_server_notification(ctx: Context, message: str, level: str = “info”):
“””
Send a real server notification using the session directly.
This is the key insight from the GitHub issue #953.
“””
try:
# Access the session directly through the context
session = ctx.session

```
    # Create notification data
    notification_data = {
        "message": message,
        "level": level,
        "timestamp": datetime.now().isoformat(),
        "server": "NotificationServer"
    }
    
    # Send progress notification (which appears as a server notification)
    await session.send_progress_notification(
        progress_token=f"notification_{int(time.time())}",
        progress=1.0,
        total=1.0,
        message=f"[{level.upper()}] {message}",
        related_request_id=ctx.request_id
    )
    
    print(f"📡 Server notification sent: {message}")
    return True
    
except Exception as e:
    print(f"❌ Failed to send notification: {e}")
    return False
```

async def broadcast_to_all_sessions(message: str, level: str = “info”):
“””
Broadcast a notification to all active sessions.
This requires tracking sessions globally.
“””
print(f”📡 Broadcasting to {len(_active_sessions)} sessions: {message}”)

```
for session_id, session_data in _active_sessions.items():
    try:
        session = session_data["session"]
        
        # Send notification to this session
        await session.send_progress_notification(
            progress_token=f"broadcast_{int(time.time())}",
            progress=1.0,
            total=1.0,
            message=f"[BROADCAST] {message}",
            related_request_id=session_data.get("last_request_id", "broadcast")
        )
        
    except Exception as e:
        print(f"❌ Failed to broadcast to session {session_id}: {e}")
```

# === LIFECYCLE MANAGEMENT ===

@mcp.request_handler()
async def handle_initialize(ctx: Context, **kwargs):
“”“Handle initialization and track sessions”””
session_id = f”session_{len(*active_sessions) + 1}*{int(time.time())}”

```
_active_sessions[session_id] = {
    "session": ctx.session,
    "connected_at": datetime.now(),
    "session_id": session_id,
    "last_request_id": ctx.request_id
}

print(f"🟢 New session connected: {session_id}")

# Send welcome notification
await send_server_notification(ctx, f"Welcome! Session {session_id} connected.", "info")

return {"session_id": session_id, "status": "connected"}
```

# === TOOLS (Functions that can be called by AI) ===

@mcp.tool()
async def send_notification(notification: NotificationRequest, ctx: Context) -> Dict[str, Any]:
“”“Send a notification with different priority levels and optional broadcasting.”””

```
# Store the notification
notification_data = {
    "id": len(_server_data["notifications"]) + 1,
    "message": notification.message,
    "priority": notification.priority,
    "broadcast": notification.broadcast,
    "timestamp": datetime.now().isoformat()
}
_server_data["notifications"].append(notification_data)

# Update session tracking
session_id = None
for sid, sdata in _active_sessions.items():
    if sdata["session"] == ctx.session:
        session_id = sid
        sdata["last_request_id"] = ctx.request_id
        break

# Send the notification
if notification.broadcast:
    await broadcast_to_all_sessions(notification.message, notification.priority)
    await send_server_notification(ctx, f"Broadcast sent: {notification.message}", "success")
else:
    await send_server_notification(ctx, notification.message, notification.priority)

return {
    "status": "sent",
    "notification_id": str(notification_data["id"]),
    "message": notification.message,
    "priority": notification.priority,
    "broadcast": notification.broadcast,
    "session_id": session_id
}
```

@mcp.tool()
async def start_background_task(task_request: BackgroundTaskRequest, ctx: Context) -> Dict[str, str]:
“”“Start a background task that sends periodic notifications.”””

```
task_id = f"task_{len(_server_data['active_tasks']) + 1}_{int(time.time())}"

# Store task info
_server_data["active_tasks"][task_id] = {
    "name": task_request.task_name,
    "duration": task_request.duration_seconds,
    "interval": task_request.interval_seconds,
    "started_at": datetime.now().isoformat(),
    "status": "running"
}

# Start the background task
asyncio.create_task(run_background_task(task_id, task_request, ctx))

await send_server_notification(ctx, f"Background task '{task_request.task_name}' started with ID: {task_id}", "info")

return {
    "status": "started",
    "task_id": task_id,
    "task_name": task_request.task_name,
    "duration_seconds": task_request.duration_seconds
}
```

async def run_background_task(task_id: str, task_request: BackgroundTaskRequest, ctx: Context):
“”“Run a background task with periodic notifications.”””

```
start_time = time.time()
step = 0
total_steps = task_request.duration_seconds // task_request.interval_seconds

try:
    while time.time() - start_time < task_request.duration_seconds:
        step += 1
        
        # Send progress notification
        await ctx.session.send_progress_notification(
            progress_token=task_id,
            progress=step,
            total=total_steps,
            message=f"Background task '{task_request.task_name}' - Step {step}/{total_steps}",
            related_request_id=ctx.request_id
        )
        
        print(f"📊 Background task {task_id} progress: {step}/{total_steps}")
        
        # Also broadcast to all sessions
        await broadcast_to_all_sessions(
            f"Task '{task_request.task_name}' progress: {step}/{total_steps}",
            "info"
        )
        
        await asyncio.sleep(task_request.interval_seconds)
    
    # Task completed
    _server_data["active_tasks"][task_id]["status"] = "completed"
    
    await send_server_notification(ctx, f"Background task '{task_request.task_name}' completed!", "success")
    await broadcast_to_all_sessions(f"Task '{task_request.task_name}' completed!", "success")
    
except Exception as e:
    _server_data["active_tasks"][task_id]["status"] = "failed"
    await send_server_notification(ctx, f"Background task '{task_request.task_name}' failed: {e}", "error")
```

@mcp.tool()
async def long_running_operation(steps: int, ctx: Context) -> str:
“”“Simulate a long-running operation with real-time progress notifications.”””

```
operation_id = f"operation_{int(time.time())}"

await send_server_notification(ctx, f"Starting long operation with {steps} steps...", "info")

for i in range(steps):
    # Send progress using the session directly (the working approach from GitHub issue)
    await ctx.session.send_progress_notification(
        progress_token=operation_id,
        progress=i + 1,
        total=steps,
        message=f"Processing step {i + 1} of {steps}",
        related_request_id=ctx.request_id
    )
    
    print(f"📈 Operation progress: {i + 1}/{steps}")
    
    # Simulate work
    await asyncio.sleep(1)

await send_server_notification(ctx, "Long operation completed successfully!", "success")

return f"✅ Operation completed! Processed {steps} steps."
```

@mcp.tool()
def get_server_status(ctx: Context) -> Dict[str, Any]:
“”“Get current server status and statistics.”””
return {
“status”: “running”,
“active_sessions”: len(_active_sessions),
“total_notifications”: len(_server_data[“notifications”]),
“active_tasks”: len([t for t in _server_data[“active_tasks”].values() if t[“status”] == “running”]),
“current_time”: datetime.now().isoformat(),
“transport”: “streamable_http”,
“notifications_enabled”: True
}

# === RESOURCES (Data that can be accessed) ===

@mcp.resource(“server://status”)
def get_server_status_resource() -> str:
“”“Get detailed server status as a resource.”””
status = {
“server_name”: “NotificationServer”,
“active_sessions”: len(_active_sessions),
“session_details”: [
{
“session_id”: sdata[“session_id”],
“connected_at”: sdata[“connected_at”].isoformat()
}
for sdata in _active_sessions.values()
],
“total_notifications”: len(_server_data[“notifications”]),
“active_tasks”: _server_data[“active_tasks”],
“current_time”: datetime.now().isoformat()
}

```
import json
return json.dumps(status, indent=2, default=str)
```

@mcp.resource(“notifications://recent”)
def get_recent_notifications() -> str:
“”“Get recent notifications from the server.”””
if not _server_data[“notifications”]:
return “No notifications found”

```
recent = _server_data["notifications"][-10:]  # Last 10 notifications
notification_list = []

for notif in recent:
    notification_list.append(
        f"[{notif['priority'].upper()}] {notif['message']} "
        f"(ID: {notif['id']}, Time: {notif['timestamp']}, Broadcast: {notif.get('broadcast', False)})"
    )

return "Recent Notifications:\n" + "\n".join(notification_list)
```

# === PROMPTS (Reusable templates) ===

@mcp.prompt()
def notification_demo_prompt() -> str:
“”“Generate a prompt for testing server notifications.”””
return “”“Please test the server notification system by:
1. Calling send_notification with a simple message
2. Starting a background task that runs for 10 seconds
3. Running a long operation with 5 steps
4. Sending a broadcast notification to all connected sessions

```
You should see real-time notifications appear as the operations progress!"""
```

# === MAIN EXECUTION ===

if **name** == “**main**”:
print(”=” * 60)
print(“📡 FastMCP Server with Real Server Notifications”)
print(”=” * 60)
print()
print(“🔑 Key Features:”)
print(”  📋 Real server notifications via session.send_progress_notification()”)
print(”  🔄 Background tasks with progress updates”)
print(”  📡 Broadcasting to all connected sessions”)
print(”  📊 Session tracking and management”)
print()
print(“Available Tools:”)
print(”  📤 send_notification - Send notifications with priority levels”)
print(”  🚀 start_background_task - Launch background tasks with periodic updates”)
print(”  ⏳ long_running_operation - Simulate operations with progress”)
print(”  📊 get_server_status - Get current server statistics”)
print()
print(“Resources:”)
print(”  📋 server://status - Detailed server status”)
print(”  📜 notifications://recent - Recent notification history”)
print()
print(“Starting server with Streamable HTTP transport…”)
print(“🌐 Server will be available at http://localhost:8000/mcp”)
print()

```
try:
    # Run the server with Streamable HTTP transport
    mcp.run(
        transport="streamable-http",
        host="localhost",
        port=8000
    )
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
except Exception as e:
    print(f"❌ Server error: {e}")
```
