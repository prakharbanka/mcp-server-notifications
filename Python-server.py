#!/usr/bin/env python3
“””
Production Python REPL FastAPI Server
Designed for LLM integration with proper sandboxing and concurrent execution
“””

import asyncio
import os
import sys
import tempfile
import subprocess
import resource
import signal
import shutil
import uuid
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging

logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’
)
logger = logging.getLogger(**name**)

class CodeExecutionRequest(BaseModel):
“”“Request model for code execution”””
code: str = Field(…, description=“Python code to execute”)
timeout: Optional[int] = Field(default=5, ge=1, le=30, description=“Execution timeout in seconds”)

class CodeExecutionResponse(BaseModel):
“”“Response model for code execution”””
status: str = Field(…, description=“success or error”)
stdout: str = Field(default=””, description=“Standard output”)
stderr: str = Field(default=””, description=“Standard error output”)
execution_time: str = Field(…, description=“Execution time”)
exit_code: Optional[int] = Field(default=None, description=“Process exit code”)
error: Optional[str] = Field(default=None, description=“Error message if any”)
execution_id: str = Field(…, description=“Unique execution identifier”)

class HealthResponse(BaseModel):
“”“Health check response”””
status: str
python_version: str
available_packages: list
server_info: dict

class PythonREPLSandbox:
“””
Production-grade Python REPL with process isolation and resource limits
“””

```
def __init__(self):
    self.base_env_path = Path("/opt/python-repl-env")
    self.temp_base = Path("/tmp/python-repl")
    self.temp_base.mkdir(exist_ok=True)
    
    # Thread pool for concurrent execution
    self.executor = ThreadPoolExecutor(max_workers=10)
    
    # Ensure base environment exists
    self.ensure_base_environment()
    
def ensure_base_environment(self):
    """Ensure base Python environment exists with required packages"""
    if not self.base_env_path.exists():
        logger.info("Creating base Python environment...")
        # In production, this would be pre-built in Docker image
        subprocess.run([
            sys.executable, "-m", "venv", str(self.base_env_path)
        ], check=True)
        
        # Install common packages
        pip_path = self.base_env_path / "bin" / "pip"
        packages = [
            "numpy", "pandas", "matplotlib", "scipy", "requests",
            "scikit-learn", "seaborn", "plotly", "sympy", 
            "statsmodels", "pillow", "beautifulsoup4", "lxml"
        ]
        
        try:
            subprocess.run([
                str(pip_path), "install", "--no-cache-dir"
            ] + packages, check=True, timeout=300)
            logger.info(f"Installed packages: {', '.join(packages)}")
        except subprocess.TimeoutExpired:
            logger.warning("Package installation timed out, using base Python only")

def create_execution_environment(self, execution_id: str) -> Path:
    """Create isolated execution directory"""
    exec_dir = self.temp_base / f"exec_{execution_id}"
    exec_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (exec_dir / "workspace").mkdir(exist_ok=True)
    
    return exec_dir

def set_process_limits(self):
    """Set resource limits for the subprocess"""
    try:
        # Memory limit: 256MB
        memory_limit = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # CPU time limit: 10 seconds (wall clock handled separately)
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        
        # Limit number of processes (prevent fork bombs)
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        
        # Limit file size: 10MB
        file_limit = 10 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        
        # Limit number of open files
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        
    except Exception as e:
        logger.warning(f"Could not set all resource limits: {e}")

def create_restricted_code(self, user_code: str) -> str:
    """Wrap user code with safety restrictions"""
    restricted_code = f'''
```

import sys
import os

# Restrict dangerous operations

class RestrictedBuiltins:
def **init**(self, safe_builtins):
self.safe_builtins = safe_builtins

```
def __getattr__(self, name):
    if name in self.safe_builtins:
        return self.safe_builtins[name]
    raise NameError(f"name '{{name}}' is not defined")
```

# Safe builtins list

safe_builtins = {{
‘abs’, ‘all’, ‘any’, ‘bin’, ‘bool’, ‘chr’, ‘dict’, ‘dir’,
‘enumerate’, ‘filter’, ‘float’, ‘format’, ‘frozenset’, ‘getattr’,
‘hasattr’, ‘hash’, ‘hex’, ‘id’, ‘int’, ‘isinstance’, ‘issubclass’,
‘iter’, ‘len’, ‘list’, ‘map’, ‘max’, ‘min’, ‘next’, ‘oct’,
‘ord’, ‘pow’, ‘print’, ‘range’, ‘repr’, ‘reversed’, ‘round’,
‘set’, ‘setattr’, ‘slice’, ‘sorted’, ‘str’, ‘sum’, ‘tuple’,
‘type’, ‘vars’, ‘zip’, ‘divmod’, ‘Exception’, ‘ValueError’,
‘TypeError’, ‘KeyError’, ‘IndexError’, ‘AttributeError’
}}

# Create restricted builtins

original_builtins = **builtins**
safe_builtins_dict = {{name: getattr(original_builtins, name)
for name in safe_builtins if hasattr(original_builtins, name)}}

# Override dangerous functions

def restricted_import(name, *args, **kwargs):
allowed_modules = {{
‘math’, ‘random’, ‘datetime’, ‘json’, ‘collections’, ‘itertools’,
‘functools’, ‘operator’, ‘statistics’, ‘decimal’, ‘fractions’,
‘numpy’, ‘pandas’, ‘matplotlib’, ‘scipy’, ‘sklearn’, ‘seaborn’,
‘plotly’, ‘sympy’, ‘statsmodels’, ‘PIL’, ‘bs4’, ‘requests’
}}

```
if name.split('.')[0] not in allowed_modules:
    raise ImportError(f"Import of '{{name}}' is not allowed")

return original_builtins['__import__'](name, *args, **kwargs)
```

safe_builtins_dict[’**import**’] = restricted_import

# Apply restrictions

**builtins** = safe_builtins_dict

# User code starts here

try:
{user_code}
except KeyboardInterrupt:
print(“Execution interrupted”)
sys.exit(1)
except Exception as e:
import traceback
print(f”Error: {{e}}”, file=sys.stderr)
traceback.print_exc(file=sys.stderr)
sys.exit(1)
‘’’
return restricted_code

```
async def execute_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
    """Execute Python code in isolated environment"""
    execution_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Starting execution {execution_id}")
    
    try:
        # Create execution environment
        exec_dir = self.create_execution_environment(execution_id)
        
        # Write code to file
        code_file = exec_dir / "user_code.py"
        restricted_code = self.create_restricted_code(code)
        code_file.write_text(restricted_code)
        
        # Python interpreter from base environment
        python_path = self.base_env_path / "bin" / "python"
        if not python_path.exists():
            python_path = sys.executable  # Fallback to system Python
        
        # Execute in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._execute_subprocess,
            str(python_path),
            str(code_file),
            str(exec_dir / "workspace"),
            timeout
        )
        
        execution_time = time.time() - start_time
        result['execution_time'] = f"{execution_time:.3f}s"
        result['execution_id'] = execution_id
        
        logger.info(f"Completed execution {execution_id} in {execution_time:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"Execution {execution_id} failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'stdout': '',
            'stderr': '',
            'execution_time': f"{time.time() - start_time:.3f}s",
            'execution_id': execution_id
        }
    finally:
        # Cleanup
        try:
            if 'exec_dir' in locals():
                shutil.rmtree(exec_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup failed for {execution_id}: {e}")

def _execute_subprocess(self, python_path: str, code_file: str, 
                      workspace_dir: str, timeout: int) -> Dict[str, Any]:
    """Execute subprocess with proper isolation (runs in thread pool)"""
    try:
        # Start process with restrictions
        process = subprocess.Popen([
            python_path, code_file
        ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workspace_dir,
            preexec_fn=self.set_process_limits,
            text=True,
            env={
                'PATH': '/usr/bin:/bin',  # Minimal PATH
                'PYTHONPATH': '',
                'HOME': workspace_dir,
                'TMPDIR': workspace_dir,
            }
        )
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            return {
                'status': 'success' if process.returncode == 0 else 'error',
                'stdout': stdout,
                'stderr': stderr,
                'exit_code': process.returncode
            }
            
        except subprocess.TimeoutExpired:
            # Kill process and children
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.kill()
                parent.kill()
            except:
                pass
            
            process.kill()
            
            return {
                'status': 'error',
                'error': 'Execution timeout',
                'stdout': '',
                'stderr': f'Process killed after {timeout} seconds',
                'exit_code': -1
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'stdout': '',
            'stderr': '',
            'exit_code': -1
        }
```

# Initialize sandbox

sandbox = PythonREPLSandbox()

# FastAPI app

app = FastAPI(
title=“Python REPL Server”,
description=“Production-grade Python REPL with sandboxing for LLM integration”,
version=“1.0.0”
)

# CORS middleware

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

@app.post(”/execute”, response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
“”“Execute Python code in sandboxed environment”””
if not request.code.strip():
raise HTTPException(status_code=400, detail=“Code cannot be empty”)

```
if len(request.code) > 50000:  # 50KB limit
    raise HTTPException(status_code=400, detail="Code too large")

try:
    result = await sandbox.execute_code(request.code, request.timeout)
    return CodeExecutionResponse(**result)

except Exception as e:
    logger.error(f"Execution failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/health”, response_model=HealthResponse)
async def health_check():
“”“Health check endpoint”””
try:
# Test basic execution
test_result = await sandbox.execute_code(“print(‘health check’)”, timeout=2)

```
    # Get available packages
    packages_result = await sandbox.execute_code("""
```

import sys
import pkg_resources
packages = [pkg.project_name for pkg in pkg_resources.working_set]
print(sorted(packages)[:20])  # First 20 packages
“””, timeout=5)

```
    available_packages = []
    if packages_result['status'] == 'success':
        try:
            # Extract package list from stdout
            import ast
            packages_str = packages_result['stdout'].strip()
            if packages_str.startswith('[') and packages_str.endswith(']'):
                available_packages = ast.literal_eval(packages_str)
        except:
            available_packages = ["numpy", "pandas", "matplotlib"]  # Default
    
    return HealthResponse(
        status="healthy" if test_result['status'] == 'success' else "degraded",
        python_version=sys.version,
        available_packages=available_packages,
        server_info={
            "base_env": str(sandbox.base_env_path),
            "temp_dir": str(sandbox.temp_base),
            "max_workers": sandbox.executor._max_workers,
            "memory_limit": "256MB",
            "timeout_limit": "5-30s"
        }
    )

except Exception as e:
    logger.error(f"Health check failed: {e}")
    raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")
```

@app.get(”/”)
async def root():
“”“Root endpoint with API information”””
return {
“message”: “Python REPL Server”,
“version”: “1.0.0”,
“endpoints”: {
“POST /execute”: “Execute Python code”,
“GET /health”: “Health check”,
“GET /docs”: “API documentation”
},
“features”: [
“Process isolation”,
“Resource limits”,
“Concurrent execution”,
“Pre-installed ML packages”,
“LLM-friendly API”
]
}

@app.on_event(“startup”)
async def startup_event():
“”“Startup event”””
logger.info(“🐍 Python REPL Server starting…”)
logger.info(f”Base environment: {sandbox.base_env_path}”)
logger.info(f”Temp directory: {sandbox.temp_base}”)
logger.info(“✅ Server ready for code execution”)

@app.on_event(“shutdown”)
async def shutdown_event():
“”“Shutdown event”””
logger.info(“🛑 Python REPL Server shutting down…”)
sandbox.executor.shutdown(wait=True)

```
# Cleanup temp directories
try:
    shutil.rmtree(sandbox.temp_base, ignore_errors=True)
except:
    pass

logger.info("✅ Cleanup completed")
```

if **name** == “**main**”:
# Production server configuration
uvicorn.run(
“main:app”,
host=“0.0.0.0”,
port=8000,
workers=1,  # Single worker for resource management
log_level=“info”,
access_log=True
)
