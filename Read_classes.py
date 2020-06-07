def load_classes(classes_file):
    fp = open(classes_file, "r")
    classes = fp.read().split("\n")[:-1]
    return classes
