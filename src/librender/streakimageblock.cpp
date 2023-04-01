#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/render/streakimageblock.h>

// use Enoki complex numbers instead of std ones
#include <enoki/complex.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT
StreakImageBlock<Float, Spectrum>::StreakImageBlock(
    const ScalarVector2i &size, int32_t time, int32_t freq_resolution,
    float lo_fbound, float hi_fbound, float exposure_time,
    float time_offset, size_t channel_count, bool freq_transform, const ReconstructionFilter *filter,
    const ReconstructionFilter *time_filter, bool warn_negative,
    bool warn_invalid, bool border, bool normalize)
    : m_offset(0), m_size(0), m_depth(0), m_time(time), m_freq_resolution(freq_resolution), 
      m_lo_fbound(lo_fbound), m_hi_fbound(hi_fbound), m_exposure_time(exposure_time),
      m_time_offset(time_offset), m_channel_count((uint32_t) channel_count),
      m_freq_transform(freq_transform), m_filter(filter), m_time_filter(time_filter), m_weights_x(nullptr),
      m_weights_y(nullptr), m_warn_negative(warn_negative),
      m_warn_invalid(warn_invalid), m_normalize(normalize), m_freqs(m_freq_resolution, 0) {

    m_border_size = (uint32_t)((filter != nullptr && border) ? filter->border_size() : 0);
    m_time_border_size = (uint32_t)((time_filter != nullptr && border) ? time_filter->border_size() : 0);

    if (filter) {
        // Temporary buffers used in put()
        int filter_size = (int) std::ceil(2 * filter->radius()) + 1;
        m_weights_x     = new Float[2 * filter_size];
        m_weights_y     = m_weights_x + filter_size;
    }

    if ( m_freq_transform ) {
        for ( int i = 0; i < m_freq_resolution; i++ ) {
            m_freqs[i] = m_lo_fbound + (m_hi_fbound - m_lo_fbound) * (float)i / (float)(m_freq_resolution - 1);
        }
    }
    // TODO(jorge): initialize also the time_filter

    // set size depending on the frequency resolution if freq transform is enabled
    set_size(size, (freq_transform ? freq_resolution : time));
}

MTS_VARIANT StreakImageBlock<Float, Spectrum>::~StreakImageBlock() {
    /* Note that m_weights_y points to the same array as
       m_weights_x, so there is no need to delete it. */
    if (m_weights_x)
        delete[] m_weights_x;
}

MTS_VARIANT void StreakImageBlock<Float, Spectrum>::clear() {
    size_t size = m_channel_count * length() * hprod(m_size + 2 * m_border_size);
    if constexpr (!is_cuda_array_v<Float>)
        memset(m_data.data(), 0, size * sizeof(ScalarFloat));
    else
        m_data = zero<DynamicBuffer<Float>>(size);
}

MTS_VARIANT void
StreakImageBlock<Float, Spectrum>::set_size(const ScalarVector2i &size,
                                            const int32_t depth) {
    if ((size == m_size) && (depth == m_depth))
        return;
    m_size = size;
    m_depth = depth;
        
    m_data = empty<DynamicBuffer<Float>>(m_channel_count * depth *
                                         hprod(size + 2 * m_border_size));
}
MTS_VARIANT void
StreakImageBlock<Float, Spectrum>::put(const StreakImageBlock *block) {
    ScopedPhase sp(ProfilerPhase::ImageBlockPut);

    if (unlikely(block->channel_count() != channel_count()))
        Throw("ImageBlock::put(): mismatched channel counts!");

    if (unlikely(block->length() != length() ||
                 block->exposure_time() != exposure_time()))
        Throw("ImageBlock::put(): mismatched length or exposure_time!");

    ScalarVector2i source_size  = block->size() + 2 * block->border_size(),
                   target_size  = size()        + 2 * border_size();

    ScalarPoint2i source_offset = block->offset() - block->border_size(),
                  target_offset = offset()        - border_size();

    uint32_t source_length = block->length(),
             target_length = length();

    if constexpr (is_cuda_array_v<Float> || is_diff_array_v<Float>) {
        accumulate_3d<Float &, const Float &>(
            block->data(), source_size, source_length,
            data(), target_size, target_length,
            ScalarVector2i(0), source_offset - target_offset,
            source_size, channel_count());
    } else {
        accumulate_3d(
            block->data().data(), source_size, source_length,
            data().data(), target_size, target_length,
            ScalarVector2i(0), source_offset - target_offset,
            source_size, channel_count());
    }
}

// compute one term of the sum approximation of the integral Fourier Transform.
template <typename Float>
Complex<Float> ft_partial_term(Float value, Float t, Float freq) {
    Complex<Float> I(0.0f, 1.0f); 
    return value * exp(-2.0f * M_PI * I * freq * t);
}

MTS_VARIANT void
StreakImageBlock<Float, Spectrum>::put(
    const Point2f &pos_, const std::vector<FloatSample<Float>> &values) {
    ScopedPhase sp(ProfilerPhase::ImageBlockPut);
    Assert(m_filter != nullptr);
    // TODO(jorge): assert m_time_filter != nullptr and use it later

    for (const auto &radiance_sample : values) {

        std::cout << radiance_sample.opl << std::endl;
        Mask active = radiance_sample.mask;
        std::cout << "mask : " << active << std::endl;
        // Convert t to bin
        Float pos_sensor      = (radiance_sample.opl - m_time_offset) / m_exposure_time;
        Int32 pos_sensor_int = floor2int<Int32>(pos_sensor);

        // Check if all sample values are valid
        if (likely(m_warn_negative || m_warn_invalid)) {
            Mask is_valid = true;

            if (m_warn_negative) {
                for (uint32_t k = 0; k < m_channel_count; ++k)
                    is_valid &= radiance_sample.values[k] >= -1e-5f;
            }

            if (m_warn_invalid) {
                for (uint32_t k = 0; k < m_channel_count; ++k)
                    is_valid &= enoki::isfinite(radiance_sample.values[k]);
            }

            if (unlikely(any(active && !is_valid))) {
                std::ostringstream oss;
                oss << "Invalid sample value: [";
                for (uint32_t i = 0; i < m_channel_count; ++i) {
                    oss << radiance_sample.values[i];
                    if (i + 1 < m_channel_count)
                        oss << ", ";
                }
                oss << "]";
                Log(Warn, "%s", oss.str());
                active &= is_valid;
            }
        }

        // Check if pos_sensor_int is within the time range
        active &= (0 <= pos_sensor_int && pos_sensor_int < m_time);

        ScalarFloat filter_radius = m_filter->radius();
        ScalarVector2i size       = m_size + 2 * m_border_size;

        // Convert to pixel coordinates within the image block
        Point2f pos = pos_ - (m_offset - m_border_size + .5f);

        //Float freq = m_lo_fbound + (float)i * (m_hi_fbound - m_lo_fbound) / (float)m_time;

        if (filter_radius > 0.5f + math::RayEpsilon<Float>) {
            // Determine the affected range of pixels
            Point2u lo = Point2u(max(ceil2int<Point2i>(pos - filter_radius), 0)),
                    hi = Point2u(min(floor2int<Point2i>(pos + filter_radius), size - 1));

            uint32_t n = ceil2int<uint32_t>((m_filter->radius() - 2.f * math::RayEpsilon<ScalarFloat>) *2.f);

            Point2f base = lo - pos;
            for (uint32_t i = 0; i < n; ++i) {
                Point2f p = base + i;
                if constexpr (!is_cuda_array_v<Float>) {
                    m_weights_x[i] = m_filter->eval_discretized(p.x(), active);
                    m_weights_y[i] = m_filter->eval_discretized(p.y(), active);
                } else {
                    m_weights_x[i] = m_filter->eval(p.x(), active);
                    m_weights_y[i] = m_filter->eval(p.y(), active);
                }
            }

            if (unlikely(m_normalize)) {
                Float wx(0), wy(0);
                for (uint32_t i = 0; i <= n; ++i) {
                    wx += m_weights_x[i];
                    wy += m_weights_y[i];
                }
                Float factor = rcp(wx * wy);
                for (uint32_t i = 0; i <= n; ++i)
                    m_weights_x[i] *= factor;
            }

            ENOKI_NOUNROLL for (uint32_t yr = 0; yr < n; ++yr) {
                UInt32 y     = lo.y() + yr;
                Mask enabled = active && y <= hi.y();

                ENOKI_NOUNROLL for (uint32_t xr = 0; xr < n; ++xr) {
                    UInt32 x      = lo.x() + xr;
                    UInt32 offset = m_channel_count * (y * size.x() * m_time +
                                                    x * m_time + pos_sensor_int);

                    UInt32 time_offset = m_channel_count * (lo.y() * size.x() * m_time +
                                            lo.x() * m_time + pos_sensor_int);

                    Float weight  = m_weights_y[yr] * m_weights_x[xr];

                    enabled &= x <= hi.x();
                    ENOKI_NOUNROLL for (uint32_t k = 0; k < m_channel_count; ++k)

                        if ( m_freq_transform ) {

                            for ( int f = 0; f < m_freq_resolution; f++ ) {

                                UInt32 freq_offset = offset + m_channel_count * f;
                                Complex<Float> ft = ft_partial_term<Float>(radiance_sample.values[k] * weight, radiance_sample.opl / m_exposure_time, m_freqs[f]);
                                scatter_add(m_data, real(ft), freq_offset + k, enabled);
                            }
                        } else {
                        
                            scatter_add(m_data, radiance_sample.values[k] * weight, time_offset + k, enabled);
                        }
                }
            }
        } else {
            Point2u lo    = ceil2int<Point2i>(pos - .5f);
            UInt32 offset = m_channel_count * (lo.y() * size.x() * m_depth +
                                            lo.x() * m_depth);

            UInt32 time_offset = m_channel_count * (lo.y() * size.x() * m_time +
                                            lo.x() * m_time + pos_sensor_int);

            Mask enabled  = active && all(lo >= 0u && lo < size);
            ENOKI_NOUNROLL for (uint32_t k = 0; k < m_channel_count; ++k) {

                if ( m_freq_transform ) {

                    for ( int f = 0; f < m_freq_resolution; f++ ) {

                        UInt32 freq_offset = offset + m_channel_count * f;
                        Complex<Float> ft = ft_partial_term<Float>(radiance_sample.values[k], radiance_sample.opl / m_exposure_time, m_freqs[f]);
                        scatter_add(m_data, real(ft), freq_offset + k, enabled);
                    }
                } else {
                
                    scatter_add(m_data, radiance_sample.values[k], time_offset + k, enabled);
                }
            }
        }
    }
}

MTS_VARIANT DynamicBuffer<Float> StreakImageBlock<Float, Spectrum>::data(int slice_) const {
    uint32_t values_per_slice = m_channel_count * length() * width();
    uint32_t offset = values_per_slice * slice_;
    DynamicBuffer<ScalarUInt32> idx = arange<DynamicBuffer<ScalarUInt32>>(offset, offset + values_per_slice);
    DynamicBuffer<Float> slice_x_t = gather<DynamicBuffer<Float>>(m_data, idx);
    return slice_x_t;
}

MTS_VARIANT std::string StreakImageBlock<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "StreakImageBlock[" << std::endl
        << "  freq_transform = " << std::boolalpha << m_freq_transform << std::endl 
        << "  lo_fbound = " << std::boolalpha << m_lo_fbound << std::endl
        << "  hi_fbound = " << std::boolalpha << m_hi_fbound << std::endl
        << "  offset = " << m_offset << "," << std::endl
        << "  size = " << m_size << "," << std::endl
        << "  time = " << m_time << "," << std::endl
        << "  exposure_time = " << m_exposure_time << "," << std::endl
        << "  warn_negative = " << m_warn_negative << "," << std::endl
        << "  warn_invalid = " << m_warn_invalid << "," << std::endl
        << "  border_size = " << m_border_size << "," << std::endl
        << "  length = " << length() << "," << std::endl
        << "  time_border_size = " << m_time_border_size;
    if (m_filter)
        oss << "," << std::endl << "  filter = " << string::indent(m_filter);
    if (m_time_filter)
        oss << "," << std::endl << "  time_filter = " << string::indent(m_time_filter);
    oss << std::endl << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS_VARIANT(StreakImageBlock, Object)
MTS_INSTANTIATE_CLASS(StreakImageBlock)
NAMESPACE_END(mitsuba)
