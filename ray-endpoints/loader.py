"""
Endpoint loader and discovery
"""
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import List, Type
import sys

# Handle both direct execution and module import
try:
    from .base import BaseEndpoint
except ImportError:
    # If relative import fails, import directly from current directory
    _current_dir = Path(__file__).parent
    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))
    import base
    BaseEndpoint = base.BaseEndpoint

def discover_endpoints(endpoints_dir: str = None) -> List[Type[BaseEndpoint]]:
    """
    Automatically discover all endpoint classes in the endpoints directory.
    """
    if endpoints_dir is None:
        endpoints_dir = Path(__file__).parent / "endpoints"
    else:
        endpoints_dir = Path(endpoints_dir)
    
    endpoints = []
    
    # Walk through all Python files in endpoints directory
    for endpoint_file in endpoints_dir.glob("*.py"):
        if endpoint_file.name == "__init__.py":
            continue
        # Skip triton files - they're deployed separately, not via BaseEndpoint
        if endpoint_file.name.endswith("_triton.py"):
            continue
        
        try:
            # Load the module directly from file
            spec = importlib.util.spec_from_file_location(
                endpoint_file.stem, 
                endpoint_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all BaseEndpoint subclasses
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseEndpoint) and 
                    obj != BaseEndpoint):
                    endpoints.append(obj)
                    print(f"✓ Discovered endpoint: {obj.DEPLOYMENT_NAME}")
        except Exception as e:
            print(f"✗ Failed to load {endpoint_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return endpoints

def load_endpoint_from_file(file_path: str) -> Type[BaseEndpoint]:
    """
    Dynamically load an endpoint from a file path.
    """
    import importlib.util
    import sys
    
    spec = importlib.util.spec_from_file_location("endpoint_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["endpoint_module"] = module
    spec.loader.exec_module(module)
    
    # Find the endpoint class
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseEndpoint) and 
            obj != BaseEndpoint):
            return obj
    
    raise ValueError(f"No endpoint class found in {file_path}")
