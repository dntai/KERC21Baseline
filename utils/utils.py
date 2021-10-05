import configparser
import os
import time
import logging

def read_config_file(config_file):
    ''' Performs read config file and parses it.
    :param config_file: (String) the path of a .ini file.
    :return cfg: (dict) the dictionary of information in config_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    cf = configparser.ConfigParser()
    cf.read(config_file)
    cfg = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return cfg