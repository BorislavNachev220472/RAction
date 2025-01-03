import functools


def reset_input_callback(func):
    """
        Resets the input directory after the passed function finishes its execution.
        :param func: function
        :return:
    """
    @functools.wraps(func)
    def wrapper(instance, *args, **kwargs):
        print(instance.inner_directory)
        result = func(instance, *args, **kwargs)
        instance.inner_directory = ''
        return result

    return wrapper


def reset_output_callback(func):
    """
        Resets the output directory after the passed function finishes its execution.
        :param func: function
        :return:
    """
    @functools.wraps(func)
    def wrapper(instance, *args, **kwargs):
        print(instance.output_folder)
        result = func(instance, *args, **kwargs)
        print(args)
        instance.output_folder = ''
        return result

    return wrapper
