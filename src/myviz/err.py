class MyvizError(Exception):
    """ Base exception for all myviz errors """
    pass

class MissingVariableError(MyvizError):
    """ Raised when variable is missing from grid file """
    pass

class MissingFilepathError(MyvizError):
    """ Raised when a provided filepath is invalid """
    pass

class GridFileFormatError(MyvizError):
    """ Raised when a grid file doesn't match expected formats """
    pass

class TrackFileFormatError(MyvizError):
    """ Raised when a track path file doesn't match expected formats """
    pass