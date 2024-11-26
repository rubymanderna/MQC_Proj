"""
Unit and regression test for the MQC_Proj package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import MQC_Proj


def test_MQC_Proj_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "MQC_Proj" in sys.modules
