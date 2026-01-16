"""
Triton Inference Server client utilities
"""

import tritonclient.http as tritonhttp
from tritonclient import http as httpclient
from typing import Dict, Any, Optional, List
import numpy as np


class TritonClient:
    """Wrapper for Triton Inference Server client"""
    
    def __init__(self, url: str = "localhost:8000"):
        """
        Initialize Triton client
        
        Args:
            url: Triton server URL (host:port)
        """
        self.url = url
        self.client = tritonhttp.InferenceServerClient(url=url)
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Triton server is accessible"""
        if not self.client.is_server_live():
            raise RuntimeError(f"Triton server at {self.url} is not live")
        if not self.client.is_server_ready():
            raise RuntimeError(f"Triton server at {self.url} is not ready")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            return self.client.get_model_repository_index()
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}")
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for a specific model"""
        try:
            return self.client.get_model_metadata(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to get model metadata for {model_name}: {e}")
    
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference
        
        Args:
            model_name: Name of the model
            inputs: Dictionary of input name -> numpy array
            outputs: Optional list of output names to retrieve
        
        Returns:
            Dictionary of output name -> numpy array
        """
        # Prepare Triton inputs
        triton_inputs = []
        for name, data in inputs.items():
            infer_input = httpclient.InferInput(name, data.shape, self._numpy_to_triton_dtype(data.dtype))
            infer_input.set_data_from_numpy(data)
            triton_inputs.append(infer_input)
        
        # Prepare outputs
        triton_outputs = []
        if outputs:
            for output_name in outputs:
                triton_outputs.append(httpclient.InferRequestedOutput(output_name))
        else:
            # Request all outputs
            metadata = self.get_model_metadata(model_name)
            for output in metadata.get("outputs", []):
                triton_outputs.append(httpclient.InferRequestedOutput(output["name"]))
        
        # Perform inference
        response = self.client.infer(model_name, triton_inputs, outputs=triton_outputs)
        
        # Extract results
        results = {}
        for output in triton_outputs:
            output_name = output.name()
            results[output_name] = response.as_numpy(output_name)
        
        return results
    
    @staticmethod
    def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
        """Convert numpy dtype to Triton dtype string"""
        dtype_map = {
            np.bool_: "BOOL",
            np.int8: "INT8",
            np.int16: "INT16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
            np.uint16: "UINT16",
            np.uint32: "UINT32",
            np.uint64: "UINT64",
            np.float16: "FP16",
            np.float32: "FP32",
            np.float64: "FP64",
        }
        return dtype_map.get(dtype.type, "FP32")


def create_triton_client(url: str = "localhost:8000") -> TritonClient:
    """Factory function to create a Triton client"""
    return TritonClient(url)
