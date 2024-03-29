__version__ = '0.1'
__author__ = 'Arnaud de Mattia'
__all__ = ['make_survey','SurveyCatalogue','RandomCatalogue','RandomSkyCatalogue','generate_random_redshifts','DistanceToRedshiftArray','UniformDensityMask','RedshiftDensityMask','RedshiftDensityMask2D','KernelDensityMask','MeshDensityMask','CatalogueMeshDensityMask','DensityMaskChunk','GeometryMask','Cuboid','distance','cartesian_to_sky','sky_to_cartesian','cutsky_to_box','box_to_cutsky','rotation_matrix_from_vectors','setup_logging']

from make_survey import Catalogue,SurveyCatalogue,RandomCatalogue,RandomSkyCatalogue,generate_random_redshifts,DistanceToRedshiftArray
from make_survey import UniformDensityMask,RedshiftDensityMask,RedshiftDensityMask2D,KernelDensityMask,MeshDensityMask,CatalogueMeshDensityMask,DensityMaskChunk,GeometryMask,distance,cartesian_to_sky,sky_to_cartesian,cutsky_to_box,box_to_cutsky,rotation_matrix_from_vectors
from remap import Cuboid

import logging
import time

_logging_handler = None

def setup_logging(log_level="info"):
    """
    Turn on logging, with the specified level.
    Taken from nbodykit: https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py.

    Parameters
    ----------
    log_level : 'info', 'debug', 'warning'
        the logging level to set; logging below this level is ignored.
    
    """

    # This gives:
    #
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Rmax = 120

    levels = {
            "info" : logging.INFO,
            "debug" : logging.DEBUG,
            "warning" : logging.WARNING,
            }

    logger = logging.getLogger();
    t0 = time.time()


    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ]: ' % (time.time() - t0))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler()
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level])
