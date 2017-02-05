def _import_package_files():
    """ Dynamically import all the public attributes of the python modules in this
        file's directory (the package directory) and return a list of their names.
    """
    import os
    import inspect
    exports = []
    globals_, locals_ = globals(), locals()
    package_path = os.path.dirname(__file__)
    package_name = os.path.basename(package_path)

    for filename in os.listdir(package_path):
        modulename, ext = os.path.splitext(filename)
        if modulename[0] != '_' and ext in ('.py', '.pyw'):
            subpackage = '{}.{}'.format(
                package_name, modulename)  # pkg relative
            module = __import__(subpackage, globals_, locals_, [modulename])
            modict = module.__dict__
            names = (modict['__all__'] if '__all__' in modict else
                     [name for name in
                      modict if inspect.isclass(modict[name])])  # all public
            exports.extend(names)
            globals_.update((name, modict[name]) for name in names)

    return exports

__all__ = ['__all__'] + _import_package_files()
