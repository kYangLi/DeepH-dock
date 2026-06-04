from typing import Dict, List, Callable, Any, Optional


class FunctionRegistry:
    def __init__(self):
        self._function_info: Dict[str, Dict[str, Any]] = {}
        self._modules: Dict[str, Dict] = {}
    
    def register(self,
        name: Optional[str] | None = None,
        cli_name: Optional[str] | None = None,
        cli_help: Optional[str] = "This developer let the code speak for itself by writing nothing.",
        cli_args: Optional[List] | None = None,
        cli_default: bool = False,
    ):
        def decorator(func: Callable):
            # Prepare the function and module name
            func_name = name or func.__name__.replace('_', '-')
            auto_module = self._extract_module(func)
            module_func_name = f"{auto_module}.{func_name}"
            # Save the function and its info
            self._function_info[module_func_name] = {
                'func': func,
                'name': func_name,
                'module': auto_module,
                'cli_name': cli_name or func_name,
                'cli_help': cli_help or func.__doc__ or '',
                'cli_args': cli_args or [],
                'cli_default': cli_default,
            }
            # Establish the module tree
            module_parts = auto_module.split('.')
            _current = self._modules
            for part in module_parts:
                if part not in _current:
                    _current[part] = {'_functions': [], '_submodules': {}}
                if part == module_parts[-1]:
                    _current[part]['_functions'].append(func_name)
                _current = _current[part]['_submodules']
            # Return
            return func
        # Return
        return decorator
    
    def _extract_module(self, func: Callable) -> str:
        full_module = func.__module__
        parts = full_module.split('.')[1:-1] # Get rid of deepx_dock. and ._cli
        parts = [v.replace('_', '-') for v in parts]
        return '.'.join(parts) if parts else "NULL"
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        info = self._function_info.get(name, {})
        return info.copy() if info else {}
    
    def list_functions(self) -> List[str]:
        return list(self._function_info.keys())
    
    def get_module_tree(self) -> Dict:
        return dict(self._modules)
    
    def get_functions_in_module(self, module_parts: List[str]) -> List[str]:
        # Check the dict is good
        current = self._modules
        for part in module_parts:
            if part in current:
                current = current[part]['_submodules']
            else:
                return []
        # Get the target function
        parent = self._modules
        for part in module_parts[:-1]:
            if part in parent:
                parent = parent[part]['_submodules']
        if module_parts[-1] in parent:
            return list(parent[module_parts[-1]].get('_functions', set()))
        return []
    
    def get_submodules(self, module_parts: List[str]) -> List[str]:
        _current = self._modules
        # Get target module dict
        for part in module_parts:
            if part in _current:
                _current = _current[part]['_submodules']
            else:
                return []
        # Return!
        return list(_current.keys())


registry = FunctionRegistry()
register = registry.register
get_function_info = registry.get_function_info
list_functions = registry.list_functions
