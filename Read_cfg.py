def read_cfg(cfg_file):
    cfg = open(cfg_file, 'r')
    cfg = cfg.read().split('\n')
    cfg = [i.rstrip().lstrip() for i in cfg if (len(i) > 0 and i[0] != '#')]

    module = {}
    modules = []

    for line in cfg:
        if line[0] == "[":
            if len(module) != 0:
                modules.append(module)
                module = {}
            module["type"] = line[1:-1]
        else:
            key, value = line.split("=")
            module[key.rstrip()] = value.lstrip()
    modules.append(module)

    return modules[0], modules[1:]  # first one contains info about the net