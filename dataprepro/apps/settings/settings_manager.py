from importlib import import_module
from collections.abc import MutableMapping
from dataprepro.apps.settings import default_settings
from copy import deepcopy


class SettingManagers(MutableMapping):
    def __init__(self, values=None):
        self.attributes = {}
        self.set_setting(default_settings)
        self.update_values(values)

    def __setitem__(self, key, value):
        self.set(key, value)

    def set(self, key, value):
        self.attributes[key] = value

    def __getitem__(self, item):
        if item not in self:
            return None
        return self.attributes[item]

    def get(self, name, default=None):
        return self[name] if self[name] is not None else default

    def getint(self, name, default=0):
        return int(self.get(name, default))

    def getfloat(self, name, default=0.0):
        return float(self.get(name, default))

    def getbool(self, name, default=False):  # noqa
        got = self.get(name)
        try:
            return bool(int(got))
        except ValueError:
            if got in ('True', 'true', 'TRUE'):
                return True
            if got in ('False', 'false', 'FALSE'):
                return False
            raise ValueError('supported values for bool settings are (0 or 1),'
                             '(true or false),("0" or "1"),("True" or "False"),("TRUE" or "FALSE")')

    def getlist(self, name, default=None):
        value = self.get(name, default or [])
        if isinstance(value, str):
            value = value.split(',')
        return list(value)

    def __contains__(self, item):
        return item in self.attributes

    def __delitem__(self, key):
        self.delete(key)

    def delete(self, key):
        del self.attributes[key]

    def set_setting(self, module):
        if isinstance(module, str):
            module = import_module(module)
        for key in dir(module):
            if key.isupper():
                self.set(key, getattr(module, key))

    def __str__(self):
        return f'<Settings values = {self.attributes}>'

    def update_values(self, values): # noqa
        if values is not None:
            for key, val in values.items():
                self.set(key, val)

    def __len__(self):
        return len(self.attributes)

    def __iter__(self):
        return iter(self.attributes)

    __repr__ = __str__

    def copy(self):
        return deepcopy(self)


settings = SettingManagers()
if __name__ == '__main__':
    settings = SettingManagers()
    settings['PROJECT_NAME'] = 'baidu_spider'
    settings['CONCURRENCY'] = 1
    print(settings['CONCURRENCY'])
    print(len(settings))

