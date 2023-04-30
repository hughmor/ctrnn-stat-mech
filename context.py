import json
import datetime

class SimulationContext:
    def __init__(self):
        self._config = None

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            self._config = json.load(f)
        if 'date' not in self._config:
            self._config['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    def init_from_dict(self, config_dict):
        self._config = config_dict
        if 'date' not in self._config:
            self._config['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            
    def update_config(self, config_dict):
        self._config.update(config_dict)

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value

    def __delitem__(self, key):
        del self._config[key]

    def __getattr__(self, attr):            
        return self._config[attr]

    def __setattr__(self, attr, value):
        if attr == '_config':
            super().__setattr__(attr, value)
        else:
            self._config[attr] = value
    def __getstate__(self):
        return self._config

    def __setstate__(self, state):
        self._config = state

sim_context = SimulationContext()
