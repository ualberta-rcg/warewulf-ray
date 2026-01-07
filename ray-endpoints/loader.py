"""
Endpoint loader and discovery
"""
import importlib
import pkgutil
import inspect
from pathlib import Path
from typing import List, Type
from .base import BaseEndpoint

def discover_endpoints(endpoints_dir: str = None) -> List[Type[BaseEndpoint]]:
    """
    Automatically discover all endpoint classes in the endpoints directory.
    """
    if endpoints_dir is None:
        endpoints_dir = Path(__file__).parent / "endpoints"
    else:
        endpoints_dir = Path(endpoints_dir)
    
    endpoints = []
    
    # Import the endpoints package
    from . import endpoints as endpoints_pkg
    
    # Walk through all modules in endpoints directory
    for importer, modname, ispkg in pkgutil.iter_modules(
        endpoints_pkg.__path__, 
        endpoints_pkg.__name__ + "."
    ):
        if not ispkg:
            try:
                # Import the module
                module = importlib.import_module(modname)
                
                # Find all BaseEndpoint subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseEndpoint) and 
                        obj != BaseEndpoint):
                        endpoints.append(obj)
                        print(f"✓ Discovered endpoint: {obj.DEPLOYMENT_NAME}")
            except Exception as e:
                print(f"✗ Failed to load {modname}: {e}")
    
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
