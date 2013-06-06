#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'rft_test.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


import unittest
import ert.ecl.ecl as ecl
from   test_util import approx_equal, approx_equalv


RFT_file = "test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.RFT"
PLT_file = "test-data/Statoil/ECLIPSE/RFT/TEST1_1A.RFT"


class RFTTest( unittest.TestCase ):

    def loadRFT( self ):
        rftFile = ecl.EclRFTFile( RFT_file )
        for rft in rftFile:
            self.assertTrue( rft.is_RFT() )


    def loadPLT( self ):
        pltFile = ecl.EclRFTFile( PLT_file )
        self.assertTrue( pltFile[11].is_PLT() )
        

def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( RFTTest( 'loadRFT' ))
    suite.addTest( RFTTest( 'loadPLT' ))
    return suite


def test_suite( argv ):
    return fast_suite()


if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )


        
