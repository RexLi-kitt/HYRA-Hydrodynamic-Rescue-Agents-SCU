"""Match scenes to their registered contracts and workers without loading training.

The three adapters retain their validation and model release gates. The original
grid worker consumes checked data; the 100m workers revalidate the raw request.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Profile:
    name: str
    prefix: str
    contract_name: str
    runner: str

    @property
    def contract(self):
        return import_module(self.contract_name)

    def scenes(self):
        if self.name == 'original_grid':
            return [self.contract.scene_metadata(s) for s in self.contract.registered_scenes()]
        return [{k: v for k, v in row.items() if k != 'path'}
                for row in self.contract.scene_records()]

    def worker_payload(self, request, checked, run_id):
        if self.name == 'original_grid':
            return dict(**checked, run_id=run_id)
        return request


PROFILES = (
    Profile('scale100', 'scale100:', 'console_scale100_contract', 'console_scale100_runner.py'),
    Profile('signal_routes', 'signal-routes:', 'console_signal_routes_contract', 'console_signal_routes_runner.py'),
    Profile('original_grid', '', 'workbench_api', 'console_workbench_runner.py'),
)
PROFILES_BY_NAME = {profile.name: profile for profile in PROFILES}
DEFAULT_PROFILE = PROFILES_BY_NAME['original_grid']


def select_profile(request):
    if not isinstance(request, dict):
        raise ValueError('任务参数必须是对象')
    scene = request.get('scene')
    if isinstance(scene, str):
        for profile in PROFILES:
            if profile.prefix and scene.startswith(profile.prefix):
                return profile
    return DEFAULT_PROFILE


def catalog(name):
    return PROFILES_BY_NAME[name].contract.catalog()


def scenes():
    return [scene for profile in PROFILES for scene in profile.scenes()]


def history():
    return DEFAULT_PROFILE.contract.history()
