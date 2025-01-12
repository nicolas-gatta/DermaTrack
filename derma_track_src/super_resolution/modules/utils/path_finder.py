import os
import re

class PathFinder:
    """
    A utility class for managing file and directory paths within a project.

    This class offers static methods to retrieve the application's basic source directory path
    and to construct complete paths by appending specific file paths to this basic path.
    """
    _path = None

    @staticmethod
    def get_basic_path() -> str:
        """
        Retrieves the basic path of the application's source directory.

        If the basic path hasn't been set, it calculates this path by finding the directory
        of the currently executing file and adjusting it to point to the 'derma_track_src' directory.
        This path is stored and reused for subsequent calls.

        Returns:
            str: The basic path of the application's source directory.
        """

        if PathFinder._path is None:
            app_dir_os = os.path.dirname(os.path.abspath(__file__))
            # Use re.sub to correctly identify the 'derma_track_src' directory in a cross-platform manner
            PathFinder._path = re.sub(r"^(.*[\\/](derma_track_src))([\\/].*)?$", r"\1", app_dir_os)

        return PathFinder._path

    @staticmethod
    def get_complet_path(path_to_file: str) -> str:
        """
        Constructs and returns a complete path by appending the specified file path
        to the basic path of the application's source directory.

        Parameters:
            path_to_file (str): The file path to append to the basic path.

        Returns:
            str: The complete path combining the basic path with the specified file path.
        """

        return os.path.join(PathFinder.get_basic_path(), path_to_file)