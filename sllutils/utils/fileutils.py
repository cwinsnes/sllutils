import os


def find(root, prefix=None, suffix=None, recursive=True):
    """
    A command to go through all the files in a directory or directory tree and return a generator of all items
    matching the provided format.
    Eg: find('images_of_cats', prefix='cute', suffix='.jpg')
    should return all jpgs of cute cats from the `images_of_cats` folder and all its child folders.

    This command will not follow symbolic links.

    Args:
        root: The root folder for finding files.
        prefix: Makes all files match the specified prefix.
                If None, the filename can start with anything.
        suffix: Makes all files match the specified suffix.
                If None, the filename can end with anything.
        recursive: Causes find to search recursively throug the filesystem from root.

    Returns:
        A generator of all the paths to files that match the specified conditions.
    """
    if not prefix:
        prefix = ''
    if not suffix:
        suffix = ''
    for (root_folder, child_folders, files) in os.walk(root):
        for file_ in files:
            if file_.startswith(prefix) and file_.endswith(suffix):
                yield os.path.join(root_folder, file_)
        if not recursive:
            break
