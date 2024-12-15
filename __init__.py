"""
@author: miyuu
@title: vpred-save
@nickname: vpred-save
@description: Used to save SDXL V-Prediction models directly with correct tensors.
"""
from . import save_vpred
NODE_CLASS_MAPPINGS = {
    **save_vpred.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **save_vpred.NODE_DISPLAY_NAME_MAPPINGS,
}
