# MCP Server with Real Server Notifications - Setup Guide

This guide demonstrates how to create MCP servers that send **actual server notifications** (not just tool responses) using both the low-level MCP SDK and FastMCP with session access.

## 🔍 Key Discovery

The research revealed that **FastMCP abstracts away session access** needed for true server notifications. To send real server notifications, you need to either:

1. **Use the low-level `mcp.server.lowlevel.Server`** for full session control
1. **Access the session directly in FastMCP** via `ctx.session.send_progress_notification()`

## 📋 Requirements

- Python 3.10 or higher
- `mcp` package version 1.8.0+ (includes Streamable HTTP support)
- `uv` package manager (recommended)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init mcp-notifications-demo
cd mcp-notifications-demo

# Install MCP with CLI support
uv add "mcp[cli]>=1.8.0" pydantic
```

### 2. Choose Your Approach

#### Option A: FastMCP with Session Access (Recommended)

Save the “FastMCP with Real Server Notifications” code as `fastmcp_server.py`

#### Option B: Low-Level Server (Advanced)

Save the “Low-Level MCP Server with Real Server Notifications” code as `lowlevel_server.py`

### 3. Run the Server

```bash
# FastMCP approach (Streamable HTTP)
python fastmcp_server.py

# Low-level approach (STDIO)
python lowlevel_server.py stdio
```

### 4. Test with the Client

```bash
# Save the client code as notification_client.py

# Run notification demo
python notification_client.py

# Run interactive testing
python notification_client.py --interactive
```

## 🔑 Key Insights from Research

### What Makes Real Server Notifications Work

1. **Session Access**: You need direct access to the MCP session object
1. **Progress Notifications**: Use `session.send_progress_notification()` for server-initiated messages
1. **Streamable HTTP**: Enables dynamic connection upgrades to SSE for streaming

### The Critical Code Pattern

```python
# In FastMCP, access session through context
async def send_server_notification(ctx: Context, message: str):
    await ctx.session.send_progress_notification(
        progress_token=f"notification_{int(time.time())}",
        progress=1.0,
        total=1.0,
        message=f"[SERVER] {message}",
        related_request_id=ctx.request_id
    )
```

### Transport Protocol Details

**Streamable HTTP** enables:

- Single endpoint (`/mcp`) for all communication
- Dynamic connection upgrades to SSE for streaming
- Real-time server notifications
- Session management with optional session IDs

## 🛠️ What’s Included

### Server Features

**Real Server Notifications:**

- `send_notification()` - Send notifications with priority levels
- `start_background_task()` - Launch tasks with periodic updates
- `long_running_operation()` - Operations with progress notifications
- Automatic session tracking and broadcasting

**Notification Types:**

- **Immediate notifications** - Sent as responses to tool calls
- **Progress notifications** - Sent during long-running operations
- **Background notifications** - Sent from background tasks
- **Broadcast notifications** - Sent to all connected sessions

### Client Features

**Notification Handling:**

- Enhanced progress notification handlers
- Real-time notification display
- Progress tracking and summaries
- Interactive testing commands

## 📡 Server Notification Examples

### 1. Simple Notification

```python
await send_server_notification(ctx, "Hello from server!", "info")
```

### 2. Progress Updates

```python
await ctx.session.send_progress_notification(
    progress_token="task_123",
    progress=3,
    total=10,
    message="Processing step 3 of 10"
)
```

### 3. Background Broadcasting

```python
for session_id, session_data in active_sessions.items():
    await session_data["session"].send_progress_notification(
        progress_token="broadcast",
        progress=1.0,
        total=1.0,
        message="Server maintenance starting in 5 minutes"
    )
```

## 🧪 Testing Scenarios

### Demo Sequence

1. **Simple notification** - Basic server message
1. **Broadcast notification** - Message to all sessions
1. **Long-running operation** - Progress updates
1. **Background task** - Periodic notifications
1. **Server status** - Current state information

### Interactive Commands

```bash
# In interactive mode:
notify Hello world
broadcast Important announcement
task BackgroundDemo 10
operation 5
status
summary
```

## 🔧 Configuration Options

### FastMCP Configuration

```python
# Stateful server (maintains sessions)
mcp = FastMCP("NotificationServer", stateless_http=False)

# Stateless server (for serverless deployments)  
mcp = FastMCP("NotificationServer", stateless_http=True)

# Run with Streamable HTTP
mcp.run(transport="streamable-http", host="localhost", port=8000)
```

### Low-Level Server Configuration

```python
# Create server with notification support
server = Server("NotificationServer")

# Run with different transports
# STDIO (local process)
python server.py stdio

# SSE (requires HTTP server setup)
python server.py sse

# Streamable HTTP (modern approach)
python server.py streamable-http
```

## 📚 Architecture Comparison

### FastMCP vs Low-Level Server

|Feature              |FastMCP                    |Low-Level Server           |
|---------------------|---------------------------|---------------------------|
|**Ease of Use**      |⭐⭐⭐⭐⭐ High-level decorators|⭐⭐⭐ Manual setup required  |
|**Session Access**   |⭐⭐⭐ Via context object     |⭐⭐⭐⭐⭐ Direct access        |
|**Notifications**    |⭐⭐⭐⭐ Via `ctx.session`     |⭐⭐⭐⭐⭐ Full control         |
|**Transport Support**|⭐⭐⭐⭐⭐ All transports       |⭐⭐⭐⭐ Manual transport setup|
|**Production Ready** |⭐⭐⭐⭐⭐ Built-in features    |⭐⭐⭐ Requires more work     |

### Recommendation

- **Use FastMCP** for most applications (easier with ctx.session access)
- **Use Low-Level Server** when you need maximum control over notifications

## 🐛 Troubleshooting

### Common Issues

**Notifications not appearing:**

- Ensure you’re using `ctx.session.send_progress_notification()` not just returning values
- Check that the server is running with Streamable HTTP transport
- Verify client is properly handling progress notifications

**Session tracking problems:**

- Use stateful FastMCP (`stateless_http=False`)
- Implement proper session management in request handlers
- Track sessions globally for broadcasting

**Transport issues:**

- Streamable HTTP requires MCP SDK 1.8.0+
- SSE requires proper HTTP server setup
- STDIO only works for local process communication

### Debug Tips

```python
# Add logging to see notifications being sent
print(f"📡 Sending notification: {message}")

# Check active sessions
print(f"Active sessions: {len(active_sessions)}")

# Monitor client notification handlers
print(f"📊 Notification #{self.notification_count} received")
```

## 🚀 Next Steps

1. **Extend Notifications**: Add custom notification types and handlers
1. **Persistent Storage**: Store notification history in databases
1. **Authentication**: Add secure session management
1. **Scaling**: Deploy with load balancers and multiple instances
1. **Monitoring**: Add metrics and health checks

## 📖 Additional Resources

- [MCP Specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
- [GitHub Issue #953](https://github.com/modelcontextprotocol/python-sdk/issues/953) - Progress notification fix
- [MCP Python SDK Examples](https://github.com/modelcontextprotocol/python-sdk/tree/main/examples)

-----

**Note**: This implementation demonstrates real server notifications, not just tool call responses. The server proactively pushes notifications to connected clients via SSE streams in the Streamable HTTP protocol.
