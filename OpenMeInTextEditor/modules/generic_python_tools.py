def is_iterable(thing):
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:
        iter(thing)
    except TypeError:
        return False
    else:
        return True

def flatten(value):
    flattener = lambda *m: (i for n in m for i in (flattener(*n) if is_iterable(n) else (n,)))
    return list(flattener(value))