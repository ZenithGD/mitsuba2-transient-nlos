import math
import numpy as np
import os

import pytest
import enoki as ek
import mitsuba
import time
from dataclasses import dataclass
from numpy.random import Generator, PCG64

mitsuba.set_variant("scalar_rgb")

def check_value(im, arr, atol=1e-9):
    vals = np.array(im.data(), copy=False)\
             .reshape([im.height() + 2 * im.border_size(),
                       im.width() + 2 * im.border_size(),
                       im.time(),
                       im.channel_count()])
    ref = np.empty(shape=vals.shape)
    ref[:] = arr  # Allows to benefit from broadcasting when passing `arr`

    # Easier to read in case of assert failure
    for l in range(vals.shape[3]):
        for k in range(vals.shape[2]):
            assert ek.allclose(vals[:, :, k, l], ref[:, :, k, l], atol=atol), \
                f'Time {k} Channel {k}:\n' + str(vals[:, :, k, l]) + '\n\n' + str(ref[:, :, k, l])


def test01_construct_no_filters(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    sim = StreakImageBlock(size=[32, 32], time=10, exposure_time=2, time_offset=0, channel_count=4)
    # Methods of ImageBlock
    assert sim is not None
    assert ek.all(sim.offset() == 0)
    sim.set_offset([10, 20])
    assert ek.all(sim.offset() == [10, 20])
    assert ek.all(sim.size() == [32, 32])
    assert sim.width() == 32
    assert sim.height() == 32
    assert sim.warn_invalid()
    assert sim.warn_negative()
    assert sim.border_size() == 0  # Since there's no reconstruction filter
    assert sim.channel_count() == 4
    assert sim.data() is not None

    # Methods of StreakImageBlock
    assert sim.time() == 10
    assert sim.time_border_size() == 0  # Since there's no reconstruction filter


def test02_construct_filters(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    rfilter = load_string("""<rfilter version="2.0.0" type="gaussian">
                                <float name="stddev" value="15"/>
                             </rfilter>""")
    time_filter = load_string("""<rfilter version="2.0.0" type="gaussian">
                                <float name="stddev" value="15"/>
                             </rfilter>""")

    sim = StreakImageBlock(size=[10, 11], time=10, exposure_time=2, time_offset=0, channel_count=2,
                           filter=rfilter, time_filter=time_filter, warn_invalid=False)
    assert sim is not None
    assert sim.border_size() == rfilter.border_size()
    assert sim.time_border_size() == time_filter.border_size()
    assert sim.channel_count() == 2
    assert not sim.warn_invalid()


def test03_put_values_basic_no_filter(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    rfilter = load_string("""<rfilter version="2.0.0" type="box"/>""")
    exposure_time = 2
    sim = StreakImageBlock(size=[20, 5], time=10, exposure_time=exposure_time, time_offset=0, channel_count=2, filter=rfilter)
    sim.clear()
    check_value(sim, 0)

    sim2 = StreakImageBlock(size=sim.size(), time=sim.time(), exposure_time=2, time_offset=0,
                            channel_count=sim.channel_count(), filter=rfilter)
    sim2.clear()
    ref = 3.14 * np.arange(sim.height() * sim.width() * sim.time() * sim.channel_count()) \
                   .reshape([sim.height(), sim.width(), sim.time(), sim.channel_count()])

    for x in range(sim.height()):
        for y in range(sim.width()):
            for z in range(sim.time()):
                sim2.put([y+0.5, x+0.5], [(z*exposure_time, ref[x, y, z, :], True)])

    check_value(sim2, ref)


def test04_put_image_block(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    # Define sim Streak Imake Block as empty
    rfilter = load_string("""<rfilter version="2.0.0" type="box"/>""")
    exposure_time = 2
    sim = StreakImageBlock(size=[20, 5], time=10, exposure_time=exposure_time, time_offset=0, channel_count=2, filter=rfilter)
    sim.clear()
    check_value(sim, 0)

    # Define sim2 Streak Image Block as empty
    sim2 = StreakImageBlock(size=sim.size(), time=sim.time(), exposure_time=sim.exposure_time(), time_offset=sim.time_offset(),
                            channel_count=sim.channel_count(), filter=rfilter)
    sim2.clear()
    # Copy values into sim2
    ref = 3.14 * np.arange(sim.height() * sim.width() * sim.time() * sim.channel_count()) \
        .reshape([sim.height(), sim.width(), sim.time(), sim.channel_count()])

    for x in range(sim.height()):
        for y in range(sim.width()):
            for z in range(sim.time()):
                sim2.put([y+0.5, x+0.5], [(z*exposure_time, ref[x, y, z, :], True)])

    check_value(sim2, ref)

    # Copy sim2 into sim
    for i in range(1):
        sim.put(sim2)
        check_value(sim, (i+1) * ref)


def test05_put_values_basic(variant_scalar_rgb):
    from mitsuba.core import srgb_to_xyz
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    # Recall that we must pass a reconstruction filter to use the `put` methods.
    rfilter = load_string("""<rfilter version="2.0.0" type="box">
                                <float name="radius" value="0.4"/>
                             </rfilter>""")
    exposure_time = 2
    sim = StreakImageBlock(size=[20, 5], time=10, exposure_time=exposure_time, time_offset=0, channel_count=5, filter=rfilter)
    sim.clear()

    # From a spectrum & alpha value
    border = sim.border_size()
    ref = np.zeros(shape=(sim.height() + 2 * border,
                          sim.width() + 2 * border,
                          sim.time(),
                          3 + 1 + 1))
    for i in range(border, sim.height() + border):
        for j in range(border, sim.width() + border):
            for k in range(0, sim.time()):
                spectrum = np.random.uniform(size=(3,))
                ref[i, j, k, :3] = srgb_to_xyz(spectrum)
                ref[i, j, k, 3] = 1  # Alpha
                ref[i, j, k, 4] = 1  # Weight
                # To avoid the effects of the reconstruction filter (simpler test),
                # we'll just add one sample right in the center of each pixel.
                sim.put([j + 0.5, i + 0.5], [], [(k*exposure_time, spectrum, True)], alpha=1.0)

    check_value(sim, ref, atol=1e-6)

def test06_put_with_filter(variant_scalar_rgb):
    from mitsuba.core import srgb_to_xyz
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    """The previous tests used a very simple box filter, parametrized so that
    it essentially had no effect. In this test, we use a more realistic
    Gaussian reconstruction filter, with non-zero radius."""

    rfilter = load_string("""<rfilter version="2.0.0" type="gaussian">
            <float name="stddev" value="0.5"/>
        </rfilter>""")

    size = [12, 12]
    time = 10
    exposure_time = 1
    sim = StreakImageBlock(size, time=time, exposure_time=exposure_time, time_offset=0, channel_count=5, filter=rfilter)
    sim.clear()

    positions = np.array([
        [5, 6], [0, 1], [5, 6], [1, 11], [11, 11],
        [0, 1], [2, 5], [4, 1], [0, 11], [5, 4]
    ], dtype=np.float)
    n = positions.shape[0]
    positions += np.random.uniform(size=positions.shape, low=0, high=0.95)

    spectra = np.arange(n * 3).reshape((n, 3))
    alphas = np.ones(shape=(n,))

    radius = int(math.ceil(rfilter.radius()))
    border = sim.border_size()

    ref = np.zeros(shape=(sim.height() + 2 * border,
                          sim.width() + 2 * border,
                          sim.time(),
                          3 + 1 + 1))

    for i in range(n):
        time_random = np.random.randint(0, time)
        print(time_random)
        # -- Scalar `put`
        sim.put(positions[i, :], [], [(int(time_random * exposure_time), spectra[i, :], True)], alpha=1.0)

        # Fractional part of the position
        offset = positions[i, :] - positions[i, :].astype(np.int)

        # -- Reference
        # Reconstruction window around the pixel position
        pos = positions[i, :] - 0.5 + border
        lo  = np.ceil(pos - radius).astype(np.int)
        hi  = np.floor(pos + radius).astype(np.int)

        for dy in range(lo[1], hi[1] + 1):
            for dx in range(lo[0], hi[0] + 1):
                r_pos = np.array([dx, dy])
                w_pos = r_pos - pos

                if (np.any(r_pos < 0) or np.any(r_pos >= ref.shape[:2])):
                    continue

                weight = rfilter.eval_discretized(w_pos[0]) * rfilter.eval_discretized(w_pos[1])

                xyz = srgb_to_xyz(spectra[i, :])
                ref[r_pos[1], r_pos[0], time_random, :3] += weight * xyz
                ref[r_pos[1], r_pos[0], time_random,  3] += weight * 1  # Alpha
                ref[r_pos[1], r_pos[0], time_random,  4] += weight * 1  # Weight

    check_value(sim, ref, atol=1e-6)


def test07_data_slice(variant_scalar_rgb):
    from mitsuba.core import srgb_to_xyz
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    # Recall that we must pass a reconstruction filter to use the `put` methods.
    # Parameterized with radius < 0.5, so it does not have effect
    rfilter = load_string("""<rfilter version="2.0.0" type="box">
                                <float name="radius" value="0.4"/>
                             </rfilter>""")


    exposure_time = 2
    sim = StreakImageBlock(size=[20, 5], time=10, exposure_time=exposure_time, time_offset=0, channel_count=5, filter=rfilter)
    sim.clear()

    # Fill the StreakImageBlock with values
    ref = 3.14 * np.arange(sim.height() * sim.width() * sim.time() * sim.channel_count())\
        .reshape([sim.height(), sim.width(), sim.time(), sim.channel_count()])

    for x in range(sim.height()):
        for y in range(sim.width()):
            for z in range(sim.time()):
                sim.put([y+0.5, x+0.5], [(z*exposure_time, ref[x, y, z, :], True)])

    # Check that the values are the same
    check_value(sim, ref)

    for h in range(sim.height()):
        # Use StreakImageBlock.data(slice)
        vals = np.array(sim.data(h), copy=False)\
                 .reshape([sim.width() + 2 * sim.border_size(),
                           sim.time(),
                           sim.channel_count()])

        ref2 = np.squeeze(ref[h, :, :, :])

        # Compare values
        atol=1e-9
        # Easier to read in case of assert failure
        for l in range(vals.shape[2]):
            for k in range(vals.shape[1]):
                assert ek.allclose(vals[:, k, l], ref2[:, k, l], atol=atol), \
                    f'Height {h} Time {k} Channel {l}:\n' + str(vals[:, k, l]) + '\n\n' + str(ref2[:, k, l])

# TODO: missing test with Packet and Spectral
@dataclass
class RadianceSample:
    pos : np.array
    time : float
    spectrum : np.array
    mask : bool

def manual_fft(block, samples : list) -> np.array:
    # get properties from block

    x, y = block.size()
    freqs = block.freq_resolution()
    channels = block.channel_count()

    freq_array = np.fft.fftshift(np.fft.fftfreq(freqs, block.exposure_time()))

    print(freq_array)

    res = np.zeros((x, y, freqs, channels), dtype=np.complex128)

    for sample in samples:
        # get position in block
        p = np.floor(sample.pos).astype(np.int32)

        if ( p[0] >= x or p[1] >= y ):
            continue

        # for all frequencies accumulate transform value
        for i, f in enumerate(freq_array):
            for j, s in enumerate(sample.spectrum):
                res[p[0], p[1], i, j] += sample.spectrum[len(sample.spectrum) - j - 1] * np.exp(-2.0 * np.pi * 1j * f * sample.time)

    return res

def test08_freq_streakimageblock_one_sample(variant_scalar_rgb):
    from mitsuba.core import srgb_to_xyz
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    # Recall that we must pass a reconstruction filter to use the `put` methods.
    rfilter = load_string("""<rfilter version="2.0.0" type="box">
                                <float name="radius" value="0.1"/>
                             </rfilter>""")

    exposure_time = 1
    time = 100
    freq_resolution = time
    block_size = [1, 1]
    freqs = np.fft.fftfreq(n=freq_resolution, d=exposure_time)

    # frequency resolved block
    sim = StreakImageBlock(
        size=block_size,
        time=time,
        freq_resolution=freq_resolution,
        lo_fbound=freqs.min(),
        hi_fbound=freqs.max(),
        exposure_time=exposure_time,
        time_offset=0,
        channel_count=3,
        freq_transform=True,
        filter=rfilter,
        border=False
    )
    sim.clear()

    # time-resolved block
    sim_t = StreakImageBlock(
        size=block_size,
        time=time,
        freq_resolution=freq_resolution,
        lo_fbound=freqs.min(),
        hi_fbound=freqs.max(),
        exposure_time=exposure_time,
        time_offset=0,
        channel_count=3,
        freq_transform=False,
        filter=rfilter,
        border=False
    )
    sim_t.clear()

    border = sim.border_size()
    rng = Generator(PCG64())

    # add random samples
    spectrum1 = [1, 0.5, 1]

    samples = []

    i, j = 0, 0
    t = 1
    # add sample on the center of the pixel so the filter isn't applied.
    sim.put([j + 0.5, i + 0.5], [(t * exposure_time, spectrum1, True)])
    sim_t.put([j + 0.5, i + 0.5], [(t * exposure_time, spectrum1, True)])

    samples.append(RadianceSample(np.array([j + 0.5, i + 0.5]), t * exposure_time, spectrum1, True))

    sim_shape = (sim.height() + 2 * border,
                sim.width() + 2 * border,
                sim.freq_resolution(),
                3)

    sim_data = np.array(sim.data()).reshape(sim_shape)

    sim_t_shape = (sim_t.height() + 2 * border,
                sim_t.width() + 2 * border,
                sim_t.time(),
                3)

    sim_t_data = np.array(sim_t.data()).reshape(sim_t_shape)

    transformed = np.fft.fftshift(np.fft.fft(sim_t_data, axis=2, n=sim.freq_resolution()), axes=2)

    manual = manual_fft(sim, samples)

    print(f"Freqs: {freqs}")
    print("transf:", sim_data)
    print("fft: ", np.real(transformed))
    print("original: ", sim_t_data)
    print("manual:", manual)

    assert(np.allclose(sim_data, np.real(transformed), atol=5e-7))


def test09_freq_streakimageblock_full(variant_scalar_rgb):
    from mitsuba.core import srgb_to_xyz
    from mitsuba.core.xml import load_string
    from mitsuba.render import StreakImageBlock

    # Recall that we must pass a reconstruction filter to use the `put` methods.
    rfilter = load_string("""<rfilter version="2.0.0" type="box">
                                <float name="radius" value="0.1"/>
                             </rfilter>""")

    exposure_time = np.random.randint(1, 100)
    time = np.random.randint(2, 100)
    freq_resolution = time
    block_size = [1, 1]
    freqs = np.fft.fftfreq(n=freq_resolution, d=exposure_time)

    # frequency resolved block
    sim = StreakImageBlock(
        size=block_size,
        time=time,
        freq_resolution=freq_resolution,
        lo_fbound=freqs.min(),
        hi_fbound=freqs.max(),
        exposure_time=exposure_time,
        time_offset=0,
        channel_count=3,
        freq_transform=True,
        filter=rfilter,
        border=False
    )
    sim.clear()

    # time-resolved block
    sim_t = StreakImageBlock(
        size=block_size,
        time=time,
        freq_resolution=freq_resolution,
        lo_fbound=freqs.min(),
        hi_fbound=freqs.max(),
        exposure_time=exposure_time,
        time_offset=0,
        channel_count=3,
        freq_transform=False,
        filter=rfilter,
        border=False
    )
    sim_t.clear()

    border = sim.border_size()
    rng = Generator(PCG64())

    samples = []

    # add random samples
    for i in range(border, sim.height() + border):
        for j in range(border, sim.width() + border):
            for k in range(0, sim.time()):
                spectrum1 = rng.uniform(size=(3,))
                #spectrum2 = rng.uniform(size=(3,))

                #spectrum1 = [1, 0.5, 1]

                idx = k

                # add sample on the center of the pixel so the filter isn't applied.
                sim.put([j + 0.6, i + 0.6], [(k * exposure_time, spectrum1, True)])
                sim_t.put([j + 0.6, i + 0.6], [(k * exposure_time, spectrum1, True)])

                samples.append(RadianceSample(np.array([j + 0.6, i + 0.6]), k * exposure_time, spectrum1, True))

    sim_shape = (sim.height() + 2 * border,
        sim.width() + 2 * border,
        sim.freq_resolution(),
        3)

    sim_data = np.array(sim.data()).reshape(sim_shape)

    sim_t_shape = (sim_t.height() + 2 * border,
        sim_t.width() + 2 * border,
        sim_t.time(),
        3)

    sim_t_data = np.array(sim_t.data()).reshape(sim_t_shape)

    transformed = np.fft.fftshift(np.fft.fft(sim_t_data, axis=2, n=sim.freq_resolution()), axes=2)

    manual = manual_fft(sim, samples)

    print("Freqs:", freqs)
    print("transf:", sim_data)
    print("fft: ", np.real(transformed))
    print("original: ", sim_t_data)
    print("manual", manual)

    # mean error of the transformation
    mean_err = abs(np.mean(sim_data - np.real(transformed)))
    
    assert(mean_err < 1e-5)
