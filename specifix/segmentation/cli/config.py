from enum import Enum
from typing import List, Dict
from dataclasses import dataclass


class ColorMap(Enum):
    RED = '#EB3324'
    WHITE = '#FFFFFF'
    YELLOW = '#FFFE91'
    GREY = '#C0C0C0'
    DARK_GREY = '#808080'
    GREEN = '#377D22'
    BROWN = '#784315'
    PURPLE = '#741B7C'
    PINK = '#EF88BE'
    BLACK = '#000000'
    ORANGE = '#F08650'
    LIGHT_PINK = '#F5B3C0'
    LIGHT_GREEN = '#91C67B'
    LAVENDER = '#C2A5CC'
    ROYAL_BLUE = '#252958'
    BLUE = '#9FFCFD'
    GRAPE = '#604F9C'
    PEACH = '#F8BE9F'
    MAGENTA = '#D1115B'
    TEAL = '#176079'
    INDIGO = '#615F84'


@dataclass
class LabelCombined(object):
    id: int
    label: str
    color: ColorMap


@dataclass
class Label(object):
    id: int
    label: str
    color: ColorMap
    variations: List[str]

    def to_dict(self) -> Dict[str, LabelCombined]:
        return {f'{variation}.nrrd': LabelCombined(self.id, self.label, self.color) for variation in
                self.variations}


class Config(object):
    DATA_STRUCTURE = [
        Label(id=1, label="dR", color=ColorMap.LIGHT_GREEN, variations=['dR', 'p2DR2', 'R']),
        Label(id=2, label="dDR", color=ColorMap.PEACH, variations=['dDR', 'p2DU1', 'DR', 'p2DU']),
        Label(id=3, label="dDC", color=ColorMap.BLUE, variations=['dDC', 'dDRU', 'dL', 'dD', ]),
        Label(id=4, label="dDU", color=ColorMap.ORANGE, variations=['dDU1', 'dDU', 'DU', 'dLU']),
        Label(id=5, label="dU", color=ColorMap.MAGENTA, variations=['dU', 'dDU2', 'p2DU2']),
        Label(id=6, label="dVU", color=ColorMap.INDIGO, variations=['dVU', 'VU', 'dVRU']),
        Label(id=7, label="dVR", color=ColorMap.BROWN, variations=['dVR', 'dVRU', 'p2DR1', 'VR', 'p2DR']),
        Label(id=8, label="dRest", color=ColorMap.PINK,
              variations=['dC', 'dVC', 'dIntact', 'dINTACT', 'dV', 'dR-U', 'dRU', 'dRU cli']),
        Label(id=9, label="p1DR1", color=ColorMap.LAVENDER,
              variations=['p1DR1', 'p1VU2', 'PDR', 'p1DR', 'p1DR fin', 'pDR', 'pDR1', 'pVU2']),
        Label(id=10, label="p1DR2", color=ColorMap.TEAL, variations=['p1DR2', 'p2VR1', 'pDR2', 'p2VR']),
        Label(id=11, label="L", color=ColorMap.GRAPE, variations=['L', 'p1DR3', 'p1L', 'p2L', 'pDC', 'pDR3']),
        Label(id=12, label="p1DU1", color=ColorMap.ROYAL_BLUE, variations=['p1DU1', 'p1DU', 'P1DU1', 'pDU', 'pDU1']),
        Label(id=13, label="p1VU1", color=ColorMap.PURPLE, variations=['p1VU1', 'pVU', 'pVU cli', 'pVU1']),
        Label(id=14, label="p1DU2", color=ColorMap.YELLOW, variations=['p1DU2', 'p1VU2', 'p1VR2', 'pDU2']),
        Label(id=15, label="p1VR1", color=ColorMap.GREEN, variations=['p1VR1', 'p1VR', 'pVR1', 'pVR', 'pVR cli']),
        Label(id=16, label="p1VR2", color=ColorMap.RED, variations=['p1VR2', 'pU', 'pV', 'pVR2']),
        Label(id=17, label="M", color=ColorMap.GREY, variations=['M1', 'M0']),
        Label(id=18, label="unknown", color=ColorMap.WHITE, variations=['unknown']),
    ]
    VISUAL_STRUCTURE = [
        *DATA_STRUCTURE,
        Label(id=-1, label="ulna", color=ColorMap.DARK_GREY,
              variations=['Ulna broken', 'Ulna Broken', 'Ulna', 'Ulna Borken'])
    ]
    COMBINED_LABEL = 'combined'
    DEFAULT_UNKNOWN = 'unknown'
    NRRD_REGEX_PATTERN = r'\b(fin|cli|-)\b'
    NRRD_EXTENSION = '.nrrd'
    TIFF_EXTENSION = '.tiff'
    NRRD_INNER_DIRECTORY = 'nrrd'
    NIFTI_EXTENSION = '.nii.gz'

    IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    VISUAL_COMBINED_DICT: Dict[str, LabelCombined] = {}

    for label in VISUAL_STRUCTURE:
        VISUAL_COMBINED_DICT.update(label.to_dict())

    COMBINED_DICT: Dict[str, LabelCombined] = {}

    for label in DATA_STRUCTURE:
        COMBINED_DICT.update(label.to_dict())
