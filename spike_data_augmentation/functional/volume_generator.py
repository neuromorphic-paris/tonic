import numpy as np
from .utils import guess_event_ordering_numpy, is_multi_image


def calc_floor_ceil_delta(x):
    x_fl = np.floor(x + 1e-8)
    x_ce = np.ceil(x - 1e-8)
    x_ce_fake = np.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.astype(np.int64), dx_fl], [x_ce.astype(np.int64), dx_ce]


def create_update_combined_polarity(x, y, t, p, dt, vol_size):
    """ Creates the update for an event volume that contains both
    polarities in the same temporal channels.
    """
    inds = (vol_size[1] * vol_size[2]) * t + (vol_size[2]) * y + x

    vals = dt * p

    return inds, vals


def volume_numpy(
    events,
    sensor_size=(346, 260),
    ordering=None,
    num_time_bins=10,
    time_normalization_method="total",
    discrete_xy=True,
):
    """Creates an event volume out of events

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x'
                  to be in the ordering
    """
    if ordering is None:
        ordering = guess_event_ordering_numpy(events)

    assert "x" in ordering
    assert "y" in ordering
    assert "t" in ordering
    assert "p" in ordering

    x_loc = ordering.index("x")
    y_loc = ordering.index("y")
    t_loc = ordering.index("t")
    p_loc = ordering.index("p")

    # Cast to floats
    events = events.astype(float)

    vol_size = [num_time_bins, sensor_size[0], sensor_size[1]]
    volume = np.zeros(vol_size)

    if time_normalization_method == "total":
        events[:, t_loc] -= events[:, t_loc].min()
        events[:, t_loc] /= max(events[:, t_loc].max(), 1.0) / float(num_time_bins - 1)
    else:
        raise NotImplementedError()

    p = events[:, p_loc]

    if not discrete_xy:
        x_fl, x_ce = calc_floor_ceil_delta(events[:, x_loc])
        y_fl, y_ce = calc_floor_ceil_delta(events[:, y_loc])
        t_fl, t_ce = calc_floor_ceil_delta(events[:, t_loc])
        for x in [x_fl, x_ce]:
            for y in [y_fl, y_ce]:
                for t in [t_fl, t_ce]:
                    dt = x[1] * y[1] * t[1]
                    x_i, y_i, t_i = x[0], y[0], t[0]
                    inds, vals = create_update_combined_polarity(
                        x_i, y_i, t_i, p, dt, vol_size
                    )

                    np.add.at(volume.reshape(-1), inds, vals)
    else:
        t_fl, t_ce = calc_floor_ceil_delta(events[:, t_loc])
        x_i = events[:, x_loc].astype(np.int32)
        y_i = events[:, y_loc].astype(np.int32)

        for t in [t_fl, t_ce]:
            dt = t[1]
            t_i = t[0]
            inds, vals = create_update_combined_polarity(x_i, y_i, t_i, p, dt, vol_size)

            np.add.at(volume.reshape(-1), inds, vals)

    return volume
