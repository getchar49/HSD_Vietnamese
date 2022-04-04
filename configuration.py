import yaml


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        return str(self.__dict__)

    def update_dict(self, dictionary):
        self.__dict__['dictionary'] = dictionary


def get_config(config_name):
    with open("config.yaml", 'r') as stream:
        try:
            return Config(**yaml.safe_load(stream)[config_name])
        except yaml.YAMLError as exc:
            print(exc)
            return None


if __name__ == "__main__":
    print(get_config("HATESPEECH"))
