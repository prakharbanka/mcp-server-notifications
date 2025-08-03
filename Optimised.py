#!/usr/bin/env python3
“””
Optimized Python REPL FastAPI Server with Process Pool
Drop-in replacement for current implementation with same interface
“””

import asyncio
import multiprocessing as mp
import os
import sys
import signal
import time
import uuid
import io
import contextlib
import ast
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import psutil
import queue
import threading

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

class WorkerProcess:
“”“Long-running worker process for code execution”””

```
def __init__(self, worker_id: str):
    self.worker_id = worker_id
    self.input_queue = mp.Queue(maxsize=5)  # Limit queue size
    self.output_queue = mp.Queue()
    self.process = None
    self.is_healthy = True
    self.start_worker()

def start_worker(self):
    """Start the worker process"""
    self.process = mp.Process(
        target=self._worker_main,
        args=(self.worker_id, self.input_queue, self.output_queue),
        daemon=True
    )
    self.process.start()
    logger.info(f"Started worker {self.worker_id} with PID {self.process.pid}")

@staticmethod
def _worker_main(worker_id: str, input_queue: mp.Queue, output_queue: mp.Queue):
    """Main function for worker process"""
    logger.info(f"Worker {worker_id} initializing...")
    
    # Pre-import heavy packages to avoid import overhead per request
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import scipy
        import sklearn
        logger.info(f"Worker {worker_id} loaded packages successfully")
    except ImportError as e:
        logger.warning(f"Worker {worker_id} failed to load some packages: {e}")
        # Continue with basic packages
        np = None
        pd = None
        plt = None
    
    # Create safe execution environment with pre-loaded packages
    safe_globals = {
        '__builtins__': {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'getattr',
            'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass',
            'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct',
            'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round',
            'set', 'setattr', 'slice', 'sorted', 'str', 'sum', 'tuple',
            'type', 'vars', 'zip', 'divmod', 'Exception', 'ValueError',
            'TypeError', 'KeyError', 'IndexError', 'AttributeError',
            'ImportError', 'ZeroDivisionError', 'NameError'
        },
        'math': __import__('math'),
        'datetime': __import__('datetime'),
        'json': __import__('json'),
        'random': __import__('random'),
        'collections': __import__('collections'),
        'itertools': __import__('itertools'),
        'statistics': __import__('statistics'),
    }
    
    # Add pre-loaded packages if available
    if np is not None:
        safe_globals['numpy'] = np
        safe_globals['np'] = np
    if pd is not None:
        safe_globals['pandas'] = pd
        safe_globals['pd'] = pd
    if plt is not None:
        safe_globals['matplotlib'] = matplotlib
        safe_globals['plt'] = plt
    
    # Custom import function for allowed modules
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed_modules = {
            'numpy', 'pandas', 'matplotlib', 'matplotlib.pyplot', 'scipy',
            'sklearn', 'seaborn', 'plotly', 'sympy', 'statsmodels',
            'math', 'datetime', 'json', 'random', 'collections', 'itertools',
            'statistics', 'decimal', 'fractions', 'operator', 'functools'
        }
        
        if name.split('.')[0] not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed")
        
        return __import__(name, globals, locals, fromlist, level)
    
    safe_globals['__import__'] = safe_import
    
    logger.info(f"Worker {worker_id} ready for execution")
    
    # Main execution loop
    while True:
        try:
            # Get task from queue with timeout
            try:
                task = input_queue.get(timeout=60)  # 60s timeout
            except queue.Empty:
                continue  # Timeout is normal, continue waiting
            
            if task is None:  # Shutdown signal
                logger.info(f"Worker {worker_id} received shutdown signal")
                break
            
            execution_id, code, timeout_seconds = task
            start_time = time.time()
            
            # Execute code in safe environment
            result = WorkerProcess._execute_code_safely(
                code, safe_globals.copy(), timeout_seconds
            )
            
            execution_time = time.time() - start_time
            result['execution_time'] = f"{execution_time:.3f}s"
            result['execution_id'] = execution_id
            result['exit_code'] = 0 if result['status'] == 'success' else 1
            
            # Send result back
            output_queue.put(result)
            
        except Exception as e:
            logger.error(f"Worker {worker_id} unexpected error: {e}")
            # Send error result
            error_result = {
                'status': 'error',
                'error': f'Worker error: {e}',
                'stdout': '',
                'stderr': str(e),
                'execution_id': execution_id if 'execution_id' in locals() else 'unknown',
                'exit_code': 1
            }
            output_queue.put(error_result)
    
    logger.info(f"Worker {worker_id} shutting down")

@staticmethod
def _execute_code_safely(code: str, safe_globals: dict, timeout_seconds: int) -> Dict[str, Any]:
    """Execute code with timeout and output capture"""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timeout")
    
    try:
        # Validate syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'Syntax error: {e}',
                'error': str(e)
            }
        
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # Execute with captured stdout/stderr
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            
            exec(code, safe_globals)
        
        # Cancel alarm
        signal.alarm(0)
        
        return {
            'status': 'success',
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue()
        }
        
    except TimeoutError:
        signal.alarm(0)
        return {
            'status': 'error',
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue() + '\nExecution timeout',
            'error': 'Timeout'
        }
    except Exception as e:
        signal.alarm(0)
        return {
            'status': 'error',
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue() + f'\nError: {e}\n{traceback.format_exc()}',
            'error': str(e)
        }

def execute(self, code: str, timeout: int = 5) -> Dict[str, Any]:
    """Submit code for execution (blocking call)"""
    execution_id = str(uuid.uuid4())
    
    try:
        # Send task to worker
        self.input_queue.put((execution_id, code, timeout), timeout=2)
        
        # Get result with timeout
        result = self.output_queue.get(timeout=timeout + 10)
        return result
        
    except queue.Full:
        return {
            'status': 'error',
            'error': 'Worker queue full',
            'stdout': '',
            'stderr': 'Worker overloaded',
            'execution_id': execution_id,
            'exit_code': 1
        }
    except queue.Empty:
        return {
            'status': 'error',
            'error': 'Worker timeout',
            'stdout': '',
            'stderr': 'No response from worker',
            'execution_id': execution_id,
            'exit_code': 1
        }

def is_alive(self) -> bool:
    """Check if worker process is alive"""
    return self.process and self.process.is_alive()

def get_queue_size(self) -> int:
    """Get current queue size"""
    try:
        return self.input_queue.qsize()
    except:
        return 0

def shutdown(self):
    """Shutdown worker process gracefully"""
    try:
        if self.process and self.process.is_alive():
            # Send shutdown signal
            self.input_queue.put(None, timeout=1)
            
            # Wait for graceful shutdown
            self.process.join(timeout=5)
            
            # Force terminate if needed
            if self.process.is_alive():
                logger.warning(f"Force terminating worker {self.worker_id}")
                self.process.terminate()
                self.process.join(timeout=2)
            
            logger.info(f"Worker {self.worker_id} shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down worker {self.worker_id}: {e}")
```

class WorkerPool:
“”“Pool of worker processes with load balancing and health monitoring”””

```
def __init__(self, pool_size: int = None):
    if pool_size is None:
        # Use number of CPU cores, max 8 for memory efficiency
        pool_size = min(mp.cpu_count(), 8)
    
    self.pool_size = pool_size
    self.workers: List[WorkerProcess] = []
    self.current_worker = 0
    self.lock = threading.Lock()
    
    # Create worker processes
    self._create_workers()
    
    # Start health monitoring
    self.health_monitor = threading.Thread(target=self._health_monitor, daemon=True)
    self.health_monitor.start()
    
    logger.info(f"WorkerPool initialized with {pool_size} workers")

def _create_workers(self):
    """Create worker processes"""
    for i in range(self.pool_size):
        worker = WorkerProcess(f"worker-{i}")
        self.workers.append(worker)
        time.sleep(0.1)  # Small delay to stagger startup

def _health_monitor(self):
    """Background health monitoring and worker restart"""
    while True:
        try:
            time.sleep(30)  # Check every 30 seconds
            
            with self.lock:
                for i, worker in enumerate(self.workers):
                    if not worker.is_alive():
                        logger.warning(f"Restarting dead worker {worker.worker_id}")
                        
                        # Cleanup old worker
                        try:
                            worker.shutdown()
                        except:
                            pass
                        
                        # Create new worker
                        new_worker = WorkerProcess(f"worker-{i}-restart-{int(time.time())}")
                        self.workers[i] = new_worker
                        
                        logger.info(f"Replaced dead worker with {new_worker.worker_id}")
            
        except Exception as e:
            logger.error(f"Health monitor error: {e}")

def _select_best_worker(self) -> WorkerProcess:
    """Select worker with smallest queue (load balancing)"""
    with self.lock:
        # Find worker with smallest queue
        best_worker = min(self.workers, key=lambda w: w.get_queue_size() if w.is_alive() else float('inf'))
        
        if not best_worker.is_alive():
            # Fallback to round-robin if best worker is dead
            self.current_worker = (self.current_worker + 1) % len(self.workers)
            return self.workers[self.current_worker]
        
        return best_worker

async def execute(self, code: str, timeout: int = 5) -> Dict[str, Any]:
    """Execute code using best available worker"""
    worker = self._select_best_worker()
    
    # Execute in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    
    try:
        result = await loop.run_in_executor(
            None, worker.execute, code, timeout
        )
        return result
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return {
            'status': 'error',
            'error': f'Execution failed: {e}',
            'stdout': '',
            'stderr': str(e),
            'execution_id': str(uuid.uuid4()),
            'exit_code': 1,
            'execution_time': '0s'
        }

def get_stats(self) -> Dict[str, Any]:
    """Get pool statistics"""
    with self.lock:
        alive_workers = sum(1 for w in self.workers if w.is_alive())
        total_queue_size = sum(w.get_queue_size() for w in self.workers if w.is_alive())
        
        return {
            'total_workers': len(self.workers),
            'alive_workers': alive_workers,
            'total_queue_size': total_queue_size,
            'average_queue_size': total_queue_size / max(alive_workers, 1)
        }

def shutdown(self):
    """Shutdown all workers"""
    logger.info("Shutting down worker pool...")
    
    with self.lock:
        for worker in self.workers:
            worker.shutdown()
    
    logger.info("Worker pool shutdown complete")
```

class PythonREPLSandbox:
“””
Drop-in replacement for original PythonREPLSandbox using process pool
Maintains exact same interface: execute_code(code, timeout) -> dict
“””

```
def __init__(self):
    # Initialize worker pool instead of ThreadPoolExecutor
    self.worker_pool = WorkerPool(pool_size=6)  # 6 workers for good concurrency
    
    # Keep these for compatibility with health checks
    self.base_env_path = Path("/opt/python-repl-env") 
    self.temp_base = Path("/tmp/python-repl")
    
    logger.info("PythonREPLSandbox initialized with process pool")

async def execute_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute Python code - EXACT SAME INTERFACE as original
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        
    Returns:
        Dict with same structure as original implementation
    """
    start_time = time.time()
    execution_id = str(uuid.uuid4())
    
    logger.info(f"Starting execution {execution_id}")
    
    try:
        # Execute using worker pool
        result = await self.worker_pool.execute(code, timeout)
        
        # Ensure execution_time and execution_id are set
        if 'execution_time' not in result:
            execution_time = time.time() - start_time
            result['execution_time'] = f"{execution_time:.3f}s"
        
        if 'execution_id' not in result:
            result['execution_id'] = execution_id
        
        logger.info(f"Completed execution {execution_id} in {result['execution_time']}")
        return result
        
    except Exception as e:
        logger.error(f"Execution {execution_id} failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'stdout': '',
            'stderr': str(e),
            'execution_time': f"{time.time() - start_time:.3f}s",
            'execution_id': execution_id,
            'exit_code': 1
        }

def get_pool_stats(self) -> Dict[str, Any]:
    """Get worker pool statistics"""
    return self.worker_pool.get_stats()

def shutdown(self):
    """Shutdown the sandbox"""
    self.worker_pool.shutdown()
```

# Initialize sandbox (same as original)

sandbox = PythonREPLSandbox()

# FastAPI app (identical to original)

app = FastAPI(
title=“Python REPL Server”,
description=“Production-grade Python REPL with sandboxing for LLM integration”,
version=“2.0.0”  # Version bump to indicate optimization
)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

@app.post(”/execute”, response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
“”“Execute Python code in sandboxed environment - SAME INTERFACE”””
if not request.code.strip():
raise HTTPException(status_code=400, detail=“Code cannot be empty”)

```
if len(request.code) > 50000:
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
“”“Health check endpoint - enhanced with pool stats”””
try:
# Test basic execution
test_result = await sandbox.execute_code(“print(‘health check’)”, timeout=2)

```
    # Get pool statistics
    pool_stats = sandbox.get_pool_stats()
    
    # Test numpy execution
    numpy_test = await sandbox.execute_code("""
```

import numpy as np
result = np.mean([1, 2, 3, 4, 5])
print(f”numpy test: {result}”)
“””, timeout=5)

```
    return HealthResponse(
        status="healthy" if test_result['status'] == 'success' else "degraded",
        python_version=sys.version,
        available_packages=["numpy", "pandas", "matplotlib", "scipy", "sklearn"],
        server_info={
            "execution_method": "process_pool",
            "pool_stats": pool_stats,
            "numpy_test": "passed" if numpy_test['status'] == 'success' else "failed",
            "memory_limit": "per_worker",
            "timeout_limit": "5-30s"
        }
    )

except Exception as e:
    logger.error(f"Health check failed: {e}")
    raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")
```

@app.get(”/”)
async def root():
“”“Root endpoint - updated features list”””
return {
“message”: “Python REPL Server (Optimized)”,
“version”: “2.0.0”,
“endpoints”: {
“POST /execute”: “Execute Python code”,
“GET /health”: “Health check”,
“GET /docs”: “API documentation”
},
“features”: [
“Process pool execution”,
“Pre-loaded packages (numpy, pandas, etc.)”,
“Load balancing”,
“Health monitoring”,
“High concurrency support”,
“LLM-friendly API”
]
}

@app.on_event(“startup”)
async def startup_event():
“”“Startup event”””
logger.info(“🐍 Optimized Python REPL Server starting…”)
pool_stats = sandbox.get_pool_stats()
logger.info(f”Worker pool: {pool_stats[‘total_workers’]} workers”)
logger.info(“✅ Server ready for high-performance code execution”)

@app.on_event(“shutdown”)
async def shutdown_event():
“”“Shutdown event”””
logger.info(“🛑 Python REPL Server shutting down…”)
sandbox.shutdown()
logger.info(“✅ Cleanup completed”)

if **name** == “**main**”:
# Production server configuration
uvicorn.run(
“main:app”,
host=“0.0.0.0”,
port=8000,
workers=1,  # Single worker for process pool management
log_level=“info”,
access_log=True
)
