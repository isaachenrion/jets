def construct_object(key, dictionary, *args, **kwargs):
    Class = dictionary.get(key, None)
    if Class is None:
        raise ValueError('{} not found'.format(key))
    return Class(*args, **kwargs)
