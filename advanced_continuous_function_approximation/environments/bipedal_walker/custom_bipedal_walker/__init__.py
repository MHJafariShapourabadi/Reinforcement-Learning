from ._bipedal_walker_env_class import BipedalWalkerEnvClass
from ._bipedal_walker_modified_env_class import BipedalWalkerModifiedEnvClass
from ._bipedal_walker_modified_env import BipedalWalkerModified

from gymnasium.envs.registration import register

register(
    id="BipedalWalkerModified-v0",
    entry_point="advanced_continuous_function_approximation.environments.bipedal_walker.custom_bipedal_walker._bipedal_walker_modified_env:BipedalWalkerModified",
)


__all__ = ["BipedalWalkerEnvClass", "BipedalWalkerModifiedEnvClass", "BipedalWalkerModified"]