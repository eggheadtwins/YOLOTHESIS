from enum import Enum


class Location(Enum):
    INDOOR = "Indoor"
    OUTDOOR = "Outdoor"


class Weather(Enum):
    RAINY = "Rainy"
    FOG = "Fog"
    SUNNY = "Sunny"
    SNOW = "Snow"
