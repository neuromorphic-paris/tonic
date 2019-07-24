import numpy as np
from utils import guess_event_ordering_numpy


def dead_pixels_numpy(events, dead_pixels, ordering=None, sensor_size=(346, 260), seed=None):
    """ 
    Simulates dead pixels in the event stream.

    Arguments:
    - events - ndarray of shape [num_events, num_event_channels]
    - ordering - ordering of the event tuple inside of events, if None
                the system will take a guess through
                guess_event_ordering_numpy. This function requires 'p'
                to be in the ordering and requires that polarity is 
                encoded as -1 or 1
    - dead_pixels - A list of pixels [(x1,y1), (x2,y2) ... (xn, yn)] 
                    specifying the locations of the dead pixels to simulate. 
                    If a single number (n) is provided, then n random pixels will be 
                    chosen.
    - sensor_size - size of the sensor that was used [W,H]
    - seed - Seed for the random number generated (if needed)
    
    Returns:
    - events - returns events from all pixels except dead pixels
    - dead_pixels - the list of indexes for the dead pixels simulated

    Note: The algorithm currently doesn't guarantee that there will not be duplicate dead pixels in
        the randomly generated dead pixels. 
    """
    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
        assert "x" and "y" in ordering
    x_index = ordering.find("x")
    y_index = ordering.find("y")

    dead_pixel_indexes = None
    if type(dead_pixels) in [list, tuple]:
        dead_pixel_indexes = list(dead_pixels)
    elif type(dead_pixels) is int:
        # Create the requested number of dead pixels        
        np.random.seed(seed)
        dead_pixel_indexes = zip(np.random.randint(0, sensor_size[0], (dead_pixels,)), # x-values 
                                 np.random.randint(0, sensor_size[1], (dead_pixels,))) # y-values
        dead_pixel_indexes = list(dead_pixel_indexes)
    else:
        raise ValueError("Invalid parameter: dead_pixels needs to be a list/tuple or an int.")

    # Remove the events from the dead pixels
    dead_pixel_mask = np.full((events.shape[0]), False, dtype=bool)
    for x, y in dead_pixel_indexes:
        current_mask = np.logical_and(events[:, x_index] == x, events[:, y_index] == y)
        dead_pixel_mask = np.logical_or(current_mask, dead_pixel_mask)
    events = events[np.invert(dead_pixel_mask),:]
    
    # Return the events and the list of dead pixels
    return events, dead_pixel_indexes