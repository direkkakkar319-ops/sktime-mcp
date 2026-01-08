"""
Executor for sktime MCP.

Responsible for instantiating estimators, loading datasets,
and running fit/predict operations.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.runtime.handles import get_handle_manager

logger = logging.getLogger(__name__)


# Available demo datasets
# L-5: We can add more datasets here by directly wrapping on top of sktime datasets (https://www.sktime.net/en/latest/api_reference/datasets.html)
# L-6: We can also add custom datasets here
DEMO_DATASETS = {
    "airline": "sktime.datasets.load_airline",
    "longley": "sktime.datasets.load_longley",
    "lynx": "sktime.datasets.load_lynx",
    "shampoo": "sktime.datasets.load_shampoo_sales",
    "sunspots": "sktime.datasets.load_sunspot",
    "uschange": "sktime.datasets.load_uschange",
}


class Executor:
    """
    Execution runtime for sktime estimators.
    
    Handles instantiation, fitting, and prediction.
    """
    
    def __init__(self):
        self._registry = get_registry()
        self._handle_manager = get_handle_manager()
    
    def instantiate(
        self,
        estimator_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Instantiate an estimator and return a handle."""
        node = self._registry.get_estimator_by_name(estimator_name)
        if node is None:
            return {"success": False, "error": f"Unknown estimator: {estimator_name}"}
        
        try:
            instance = node.class_ref(**(params or {}))
            handle_id = self._handle_manager.create_handle(
                estimator_name=estimator_name,
                instance=instance,
                params=params or {},
            )
            return {
                "success": True,
                "handle": handle_id,
                "estimator": estimator_name,
                "params": params or {},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # L-7: We can also add custom load_dataset functions here
    def load_dataset(self, name: str) -> Dict[str, Any]:
        """Load a demo dataset."""
        if name not in DEMO_DATASETS:
            return {
                "success": False,
                "error": f"Unknown dataset: {name}",
                "available": list(DEMO_DATASETS.keys()),
            }
        
        try:
            module_path = DEMO_DATASETS[name]
            parts = module_path.rsplit(".", 1)
            module = __import__(parts[0], fromlist=[parts[1]])
            loader = getattr(module, parts[1])
            data = loader()
            
            if isinstance(data, tuple):
                y, X = data[0], data[1] if len(data) > 1 else None
            else:
                y, X = data, None
            
            return {
                "success": True,
                "name": name,
                "shape": y.shape if hasattr(y, 'shape') else len(y),
                "type": str(type(y).__name__),
                "data": y,
                "exog": X,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def fit(
        self,
        handle_id: str,
        y: Any,
        X: Optional[Any] = None,
        fh: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Fit an estimator."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}
        
        try:
            if fh is not None:
                instance.fit(y, X=X, fh=fh)
            elif X is not None:
                instance.fit(y, X=X)
            else:
                instance.fit(y)
            
            self._handle_manager.mark_fitted(handle_id)
            return {"success": True, "handle": handle_id, "fitted": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict(
        self,
        handle_id: str,
        fh: Optional[Union[int, List[int]]] = None,
        X: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate predictions."""
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {"success": False, "error": f"Handle not found: {handle_id}"}
        
        if not self._handle_manager.is_fitted(handle_id):
            return {"success": False, "error": "Estimator not fitted"}
        
        try:
            if fh is None:
                fh = list(range(1, 13))
            
            if X is not None:
                predictions = instance.predict(fh=fh, X=X)
            else:
                predictions = instance.predict(fh=fh)
            
            if isinstance(predictions, pd.Series):
                # Convert index to string to avoid JSON serialization issues with Period/DatetimeIndex
                predictions_copy = predictions.copy()
                predictions_copy.index = predictions_copy.index.astype(str)
                result = predictions_copy.to_dict()
            elif isinstance(predictions, pd.DataFrame):
                predictions_copy = predictions.copy()
                predictions_copy.index = predictions_copy.index.astype(str)
                result = predictions_copy.to_dict(orient='list')
            else:
                result = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
            
            return {"success": True, "predictions": result, "horizon": len(fh) if hasattr(fh, '__len__') else fh}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def fit_predict(
        self,
        handle_id: str,
        dataset: str,
        horizon: int = 12,
    ) -> Dict[str, Any]:
        """Convenience method: load data, fit, and predict."""
        data_result = self.load_dataset(dataset)
        if not data_result["success"]:
            return data_result
        
        y = data_result["data"]
        X = data_result.get("exog")
        fh = list(range(1, horizon + 1))
        
        fit_result = self.fit(handle_id, y, X=X, fh=fh)
        if not fit_result["success"]:
            return fit_result
        
        return self.predict(handle_id, fh=fh, X=X)
    
    # L-9: We can add more methods here to handle diverse use cases and their pipelines
    def instantiate_pipeline(
        self,
        components: List[str],
        params_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Instantiate a pipeline from a list of components.
        
        Args:
            components: List of estimator names in pipeline order
            params_list: Optional list of parameter dicts for each component
        
        Returns:
            Dictionary with success status and handle
        """
        if not components:
            return {"success": False, "error": "Pipeline cannot be empty"}
        
        # Validate the pipeline first
        from sktime_mcp.composition.validator import get_composition_validator
        validator = get_composition_validator()
        validation = validator.validate_pipeline(components)
        
        if not validation.valid:
            return {
                "success": False,
                "error": "Invalid pipeline composition",
                "validation_errors": validation.errors,
                "suggestions": validation.suggestions,
            }
        
        try:
            # If only one component, just instantiate it directly
            if len(components) == 1:
                params = params_list[0] if params_list else {}
                return self.instantiate(components[0], params)
            
            # Build the pipeline
            # Get all component nodes
            component_instances = []
            params_list = params_list or [{}] * len(components)
            
            for i, comp_name in enumerate(components):
                node = self._registry.get_estimator_by_name(comp_name)
                if node is None:
                    return {"success": False, "error": f"Unknown estimator: {comp_name}"}
                
                params = params_list[i] if i < len(params_list) else {}
                instance = node.class_ref(**params)
                component_instances.append(instance)
            
            # Determine the type of pipeline to create
            # Check if all but last are transformers
            all_transformers_except_last = all(
                self._registry.get_estimator_by_name(comp).task == "transformation"
                for comp in components[:-1]
            )
            
            final_task = self._registry.get_estimator_by_name(components[-1]).task
            
            if all_transformers_except_last and final_task == "forecasting":
                # Use TransformedTargetForecaster
                from sktime.forecasting.compose import TransformedTargetForecaster
                
                # Chain transformers if multiple
                if len(component_instances) == 2:
                    pipeline = TransformedTargetForecaster([
                        ("transformer", component_instances[0]),
                        ("forecaster", component_instances[1]),
                    ])
                else:
                    # Multiple transformers - chain them
                    from sktime.transformations.compose import TransformerPipeline
                    transformer_pipeline = TransformerPipeline([
                        (f"step_{i}", comp) for i, comp in enumerate(component_instances[:-1])
                    ])
                    pipeline = TransformedTargetForecaster([
                        ("transformers", transformer_pipeline),
                        ("forecaster", component_instances[-1]),
                    ])
            
            elif all_transformers_except_last and final_task in ("classification", "regression"):
                # Use sklearn-style Pipeline
                from sktime.pipeline import Pipeline
                pipeline = Pipeline([
                    (f"step_{i}", comp) for i, comp in enumerate(component_instances)
                ])
            
            elif all(self._registry.get_estimator_by_name(comp).task == "transformation" for comp in components):
                # All transformers - use TransformerPipeline
                from sktime.transformations.compose import TransformerPipeline
                pipeline = TransformerPipeline([
                    (f"step_{i}", comp) for i, comp in enumerate(component_instances)
                ])
            
            else:
                return {
                    "success": False,
                    "error": "Unsupported pipeline composition type",
                    "hint": "Currently supports: transformers → forecaster, transformers → classifier/regressor, or transformer chains"
                }
            
            # Create a handle for the pipeline
            pipeline_name = " → ".join(components)
            handle_id = self._handle_manager.create_handle(
                estimator_name=pipeline_name,
                instance=pipeline,
                params={"components": components, "params_list": params_list},
            )
            
            return {
                "success": True,
                "handle": handle_id,
                "pipeline": pipeline_name,
                "components": components,
                "params_list": params_list,
            }
        
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    def list_datasets(self) -> List[str]:
        """List available demo datasets."""
        return list(DEMO_DATASETS.keys())


_executor_instance: Optional[Executor] = None


def get_executor() -> Executor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = Executor()
    return _executor_instance
