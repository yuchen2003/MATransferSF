REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .multi_task.episode_runner import MTEpisodeRunner
REGISTRY["mt_episode"] = MTEpisodeRunner

from .multi_task.parallel_runner import MTParallelRunner
REGISTRY["mt_parallel"] = MTParallelRunner
