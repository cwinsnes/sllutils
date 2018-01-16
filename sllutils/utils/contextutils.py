import contextlib
import os


@contextlib.contextmanager
def redirect(channel, dest_filename):
    """
    A context manager to temporarily redirect a channel.
    Args:
        channel: The channel to redirect.
                 Must have a fileno() method available.

        dest_filename: Where to redirect the channel.
                       Must have write access to the destination.
    e.g.:
    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')

    Note:
        Code copied from: https://gist.github.com/msabramo/6040400
        Thanks to msabramo for the gist.
    """

    try:
        oldstdchannel = os.dup(channel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), channel.fileno())
        yield

    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, channel.fileno())
        if dest_file is not None:
            dest_file.close()
