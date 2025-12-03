"""Visibility calculator for celestial objects.

This module calculates where on Earth a constellation can be seen,
including optimal viewing locations and times.
"""

from datetime import datetime, timedelta
from typing import Optional

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time


def calculate_visibility_range(
    ra: float,
    dec: float,
    min_altitude: float = 10.0,
) -> dict:
    """Calculate where on Earth a constellation is visible.

    For celestial coordinates (RA, Dec), calculates the latitude range
    where the object rises above the specified minimum altitude.

    Args:
        ra: Right ascension in degrees (0-360)
        dec: Declination in degrees (-90 to 90)
        min_altitude: Minimum altitude above horizon in degrees

    Returns:
        Dictionary with viewing information:
            - min_latitude: Minimum latitude where visible (degrees)
            - max_latitude: Maximum latitude where visible (degrees)
            - optimal_latitude: Best viewing latitude (degrees)
            - circumpolar_above: Latitude above which circumpolar
            - never_rises_below: Latitude below which never visible
            - globally_visible: True if visible from all latitudes

    Example:
        >>> info = calculate_visibility_range(ra=83.8, dec=22.4)  # Orion
        >>> print(f"Visible from {info['min_latitude']:.1f}° to "
        ...       f"{info['max_latitude']:.1f}°")
        Visible from -67.6° to 90.0°
    """
    # Clamp inputs
    dec = max(-90.0, min(90.0, dec))
    min_altitude = max(0.0, min(90.0, min_altitude))

    # For an object at declination dec to reach altitude alt at latitude lat:
    # The object's maximum altitude is approximately 90° - |lat - dec|
    # when it crosses the meridian.
    #
    # Solving for when max_altitude >= min_altitude:
    # 90 - |lat - dec| >= min_altitude
    # |lat - dec| <= 90 - min_altitude
    # -90 + min_altitude <= lat - dec <= 90 - min_altitude
    # dec - 90 + min_altitude <= lat <= dec + 90 - min_altitude

    # Calculate latitude range
    min_lat = dec - (90.0 - min_altitude)
    max_lat = dec + (90.0 - min_altitude)

    # Clamp to valid latitude range
    min_lat = max(-90.0, min_lat)
    max_lat = min(90.0, max_lat)

    # Optimal viewing latitude (object passes directly overhead)
    optimal_lat = dec

    # Circumpolar: object never sets
    # An object is circumpolar when dec + lat > 90°
    # More precisely, when the object's minimum altitude > 0°
    # This happens when: lat > 90° - dec
    circumpolar_above = 90.0 - dec
    circumpolar_above = max(-90.0, min(90.0, circumpolar_above))

    # Never rises: object never reaches min_altitude
    # This happens when the object's max altitude < min_altitude
    # Or: lat < dec - (90° - min_altitude)
    never_rises_below = dec - (90.0 - min_altitude)
    never_rises_below = max(-90.0, min(90.0, never_rises_below))

    # Check if globally visible
    globally_visible = min_lat <= -90.0 and max_lat >= 90.0

    return {
        "min_latitude": min_lat,
        "max_latitude": max_lat,
        "optimal_latitude": optimal_lat,
        "circumpolar_above": circumpolar_above,
        "never_rises_below": never_rises_below,
        "globally_visible": globally_visible,
    }


def calculate_best_viewing_months(
    ra: float,
    current_month: int = None,
) -> list[str]:
    """Calculate the best months to view a constellation.

    A constellation is best viewed when it's opposite the Sun in the sky,
    which occurs roughly 6 months offset from when the Sun has the same RA.

    Args:
        ra: Right ascension in degrees (0-360)
        current_month: Current month (1-12), defaults to current month

    Returns:
        List of month names (e.g., ["October", "November", "December"])

    Example:
        >>> months = calculate_best_viewing_months(ra=83.8)  # Orion
        >>> print(months)
        ['November', 'December', 'January', 'February']
    """
    if current_month is None:
        current_month = datetime.now().month

    # The Sun's RA increases by ~30° per month (360° / 12 months)
    # Sun is at RA = 0° around March 20 (vernal equinox)
    # Best viewing is when object is opposite Sun (RA + 180°)

    # Calculate which month the Sun is at this RA
    # RA 0° → March (month 3)
    # RA increases ~30°/month
    sun_at_ra_month = 3 + (ra / 30.0)
    sun_at_ra_month = sun_at_ra_month % 12

    # Best viewing is ~6 months later (opposite the Sun)
    best_month = (sun_at_ra_month + 6) % 12

    # Return 4-month window centered on best month
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Get months from 2 months before to 1 month after best month
    viewing_months = []
    for offset in [-2, -1, 0, 1]:
        month_idx = int((best_month + offset) % 12)
        viewing_months.append(month_names[month_idx])

    return viewing_months


def calculate_meridian_transit_time(
    ra: float,
    observer_lon: float = 0.0,
    date: Optional[datetime] = None,
) -> datetime:
    """Calculate when object crosses the meridian (highest in sky).

    Args:
        ra: Right ascension in degrees (0-360)
        observer_lon: Observer longitude in degrees (-180 to 180)
        date: Date for calculation (defaults to today)

    Returns:
        Local time when object crosses meridian

    Example:
        >>> transit = calculate_meridian_transit_time(
        ...     ra=83.8, observer_lon=-74.0  # Orion from NYC
        ... )
        >>> print(transit.strftime("%H:%M"))
        21:35
    """
    if date is None:
        date = datetime.now()

    # Local Sidereal Time (LST) at meridian transit equals RA
    # LST = GST + observer_longitude
    # GST increases by ~1 degree per day (360° / 365.25 days)

    # Calculate days since J2000 epoch (January 1, 2000, 12:00 UT)
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    days_since_j2000 = (date - j2000).total_seconds() / 86400.0

    # GST at 0h UT (approximation)
    gst_0h = (280.46061837 + 360.98564736629 * days_since_j2000) % 360.0

    # LST when RA crosses meridian
    # LST = RA
    # GST + lon = RA
    # We need to find the time offset from 0h UT

    # Calculate required GST
    required_gst = (ra - observer_lon) % 360.0

    # Calculate time offset from 0h UT
    gst_offset = (required_gst - gst_0h) % 360.0

    # Convert GST degrees to hours (360° = 24 hours sidereal)
    # Sidereal day is ~23h 56m, so sidereal hour = 0.99727 solar hour
    hours_offset = (gst_offset / 360.0) * 23.9344696

    # Add to date (at 0h UT)
    transit_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
    transit_time += timedelta(hours=hours_offset)

    return transit_time


def is_visible_tonight(
    ra: float,
    dec: float,
    observer_lat: float,
    observer_lon: float,
    date: Optional[datetime] = None,
) -> bool:
    """Check if constellation is visible tonight from observer location.

    A constellation is visible if:
    1. It's above the horizon during nighttime
    2. The Sun is below -12° (astronomical twilight)

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        observer_lat: Observer latitude in degrees
        observer_lon: Observer longitude in degrees
        date: Date to check (defaults to today)

    Returns:
        True if visible tonight

    Example:
        >>> visible = is_visible_tonight(
        ...     ra=83.8, dec=22.4,  # Orion
        ...     observer_lat=40.7, observer_lon=-74.0  # NYC
        ... )
    """
    if date is None:
        date = datetime.now()

    # Create observer location
    location = EarthLocation(
        lat=observer_lat * u.deg,
        lon=observer_lon * u.deg,
        height=0 * u.m,
    )

    # Create target coordinates
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Check visibility at midnight
    midnight = date.replace(hour=0, minute=0, second=0)
    time = Time(midnight)

    # Get Sun position
    sun = get_sun(time)
    sun_altaz = sun.transform_to(AltAz(obstime=time, location=location))

    # Check if it's nighttime (Sun below -12°)
    if sun_altaz.alt.deg > -12.0:
        # Not astronomical twilight - check again 6 hours later
        time = Time(midnight + timedelta(hours=6))
        sun = get_sun(time)
        sun_altaz = sun.transform_to(AltAz(obstime=time, location=location))

        if sun_altaz.alt.deg > -12.0:
            return False

    # Check if target is above horizon during nighttime
    target_altaz = target.transform_to(AltAz(obstime=time, location=location))

    return target_altaz.alt.deg > 10.0


def get_viewing_regions(min_lat: float, max_lat: float) -> list[str]:
    """Map latitude ranges to geographic regions.

    Args:
        min_lat: Minimum viewing latitude
        max_lat: Maximum viewing latitude

    Returns:
        List of region names

    Example:
        >>> regions = get_viewing_regions(min_lat=20.0, max_lat=70.0)
        >>> print(regions)
        ['Tropics (northern)', 'Northern Temperate Zone']
    """
    regions = []

    # Define latitude zones
    zones = [
        (66.5, 90.0, "Arctic"),
        (23.5, 66.5, "Northern Temperate Zone"),
        (-23.5, 23.5, "Tropics"),
        (-66.5, -23.5, "Southern Temperate Zone"),
        (-90.0, -66.5, "Antarctic"),
    ]

    for zone_min, zone_max, zone_name in zones:
        # Check if viewing range overlaps with zone
        if max_lat >= zone_min and min_lat <= zone_max:
            # Partial overlap
            if min_lat > zone_min and max_lat < zone_max:
                regions.append(f"{zone_name} (partially)")
            # Full overlap
            elif min_lat <= zone_min and max_lat >= zone_max:
                regions.append(zone_name)
            # Northern part
            elif min_lat > zone_min:
                if "Tropics" in zone_name:
                    regions.append(f"{zone_name} (northern)")
                else:
                    regions.append(zone_name)
            # Southern part
            elif max_lat < zone_max:
                if "Tropics" in zone_name:
                    regions.append(f"{zone_name} (southern)")
                else:
                    regions.append(zone_name)
            else:
                regions.append(zone_name)

    return regions if regions else ["Global visibility"]


def get_example_cities(min_lat: float, max_lat: float) -> list[dict]:
    """Get example cities within viewing latitude range.

    Args:
        min_lat: Minimum viewing latitude
        max_lat: Maximum viewing latitude

    Returns:
        List of city dictionaries with name, lat, lon

    Example:
        >>> cities = get_example_cities(min_lat=40.0, max_lat=60.0)
        >>> for city in cities:
        ...     print(f"{city['name']}: {city['lat']:.1f}°N")
        New York: 40.7°N
        London: 51.5°N
    """
    # Major cities with coordinates
    cities_db = [
        {"name": "Reykjavik, Iceland", "lat": 64.1, "lon": -21.9},
        {"name": "Oslo, Norway", "lat": 59.9, "lon": 10.8},
        {"name": "Stockholm, Sweden", "lat": 59.3, "lon": 18.1},
        {"name": "Moscow, Russia", "lat": 55.8, "lon": 37.6},
        {"name": "London, UK", "lat": 51.5, "lon": -0.1},
        {"name": "Berlin, Germany", "lat": 52.5, "lon": 13.4},
        {"name": "Paris, France", "lat": 48.9, "lon": 2.4},
        {"name": "New York, USA", "lat": 40.7, "lon": -74.0},
        {"name": "Tokyo, Japan", "lat": 35.7, "lon": 139.7},
        {"name": "Los Angeles, USA", "lat": 34.1, "lon": -118.2},
        {"name": "Cairo, Egypt", "lat": 30.0, "lon": 31.2},
        {"name": "Miami, USA", "lat": 25.8, "lon": -80.2},
        {"name": "Hong Kong", "lat": 22.3, "lon": 114.2},
        {"name": "Mexico City, Mexico", "lat": 19.4, "lon": -99.1},
        {"name": "Mumbai, India", "lat": 19.1, "lon": 72.9},
        {"name": "Bangkok, Thailand", "lat": 13.8, "lon": 100.5},
        {"name": "Singapore", "lat": 1.3, "lon": 103.8},
        {"name": "Nairobi, Kenya", "lat": -1.3, "lon": 36.8},
        {"name": "Jakarta, Indonesia", "lat": -6.2, "lon": 106.8},
        {"name": "Lima, Peru", "lat": -12.0, "lon": -77.0},
        {"name": "São Paulo, Brazil", "lat": -23.6, "lon": -46.6},
        {"name": "Johannesburg, South Africa", "lat": -26.2, "lon": 28.0},
        {"name": "Buenos Aires, Argentina", "lat": -34.6, "lon": -58.4},
        {"name": "Sydney, Australia", "lat": -33.9, "lon": 151.2},
        {"name": "Melbourne, Australia", "lat": -37.8, "lon": 144.9},
        {"name": "Wellington, New Zealand", "lat": -41.3, "lon": 174.8},
        {"name": "Punta Arenas, Chile", "lat": -53.2, "lon": -70.9},
        {"name": "Ushuaia, Argentina", "lat": -54.8, "lon": -68.3},
    ]

    # Filter cities within latitude range
    matching_cities = [city for city in cities_db if min_lat <= city["lat"] <= max_lat]

    # Return up to 5 cities, well distributed
    if len(matching_cities) <= 5:
        return matching_cities

    # Sample evenly distributed cities
    step = len(matching_cities) / 5.0
    indices = [int(i * step) for i in range(5)]
    return [matching_cities[i] for i in indices]
