############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    sin, cos, radians, 
    zeros, sqrt, arctan, tan
)
import copy
from coordinates import geo_lat2geocentric

