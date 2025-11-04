# core/model_registry.py
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Tuple, Type

from bots.base_bot import BaseBot
from infrastructure.logger import logger

_registered_models: Dict[str, Tuple[Type[BaseBot], dict]] = {}


def auto_discover_bots():
    bots_pkg = Path(__file__).resolve().parent.parent / "bots"
    pkg_name = "bots"

    for _, module_name, _ in pkgutil.iter_modules([str(bots_pkg)]):
        if not module_name.endswith("_bot"):
            continue

        try:
            module = importlib.import_module(f"{pkg_name}.{module_name}")
        except Exception as e:
            logger.error(f"[BotRegistry] 加载模块 {module_name} 失败: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseBot) or obj is BaseBot:
                continue
            bot_class = obj
            family = getattr(bot_class, "family", "unknown")
            models_info = getattr(bot_class, "models_info", {})
            for model_id in models_info.keys():
                _registered_models[model_id] = (bot_class, {"model_name": model_id})
                logger.info(
                    f"[BotRegistry] 注册模型: {model_id} (family={family}, bot={bot_class.__name__})"
                )

    logger.info(f"[BotRegistry] 自动注册完成: {list(_registered_models.keys())}")


def get_registered_models() -> Dict[str, Tuple[Type[BaseBot], dict]]:
    if not _registered_models:
        auto_discover_bots()
    return _registered_models


class ModelRegistry:
    def __init__(self):
        self._factories: Dict[str, Tuple[Type[BaseBot], dict]] = {}
        self._instances: Dict[str, BaseBot] = {}
        self._bootstrap_all()

    def _bootstrap_all(self):
        registered = get_registered_models()
        for name, (cls, kwargs) in registered.items():
            self.register(name, cls, **kwargs)
        logger.info(f"[ModelRegistry] 自动加载模型完成: {list(self._factories.keys())}")

    def register(self, name: str, bot_cls: Type[BaseBot], **kwargs):
        self._factories[name] = (bot_cls, kwargs)
        logger.info(f"[ModelRegistry] 注册模型: {name}")

    def get_model(self, name: str) -> BaseBot:
        if name not in self._instances:
            if name not in self._factories:
                raise ValueError(f"未知模型: {name}")
            bot_cls, kwargs = self._factories[name]
            logger.info(f"[ModelRegistry] 初始化模型: {name}")
            self._instances[name] = bot_cls(**kwargs)  # type: ignore[arg-type]
        return self._instances[name]

    def get_available_models(self) -> list[dict]:
        """返回标准化模型信息列表"""

        return [
            {
                "id": name,
                "name": name,
                "family": getattr(cls, "family", "unknown"),
            }
            for name, (cls, _) in self._factories.items()
        ]


_model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
