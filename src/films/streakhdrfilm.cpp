#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/streakfilm.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/streakimageblock.h>

#include <mutex>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _film-streakhdrfilm:

High dynamic range streak film (:monosp:`streakhdrfilm`)
--------------------------------------------------------

.. pluginparameters::

* - width, height
  - |int|
  - Width and height of the camera sensor in pixels Default: 768, 576)
* - time
  - |int|
  - Time resolution of the streak film
* - exposure_time
  - |float|
  - Effective exposure time of each of the time frames (in optical distance).
* - time_offset
  - |float|
  - Time offset of the sensor (in optical distance). It can be interpreted as the minimal path length that will be stored.
* - file_format
  - |string|
  - Denotes the desired output file format. The options are :monosp:`openexr`
    (for ILM's OpenEXR format), :monosp:`rgbe` (for Greg Ward's RGBE format), or
    :monosp:`pfm` (for the Portable Float Map format). (Default: :monosp:`openexr`)
* - pixel_format
  - |string|
  - Specifies the desired pixel format of output images. The options are :monosp:`luminance`,
    :monosp:`luminance_alpha`, :monosp:`rgb`, :monosp:`rgba`, :monosp:`xyz` and :monosp:`xyza`.
    (Default: :monosp:`rgba`)
* - component_format
  - |string|
  - Specifies the desired floating  point component format of output images. The options are
    :monosp:`float16`, :monosp:`float32`, or :monosp:`uint32`. (Default: :monosp:`float16`)
* - crop_offset_y, crop_offset_y, crop_width, crop_height
  - |int|
  - These parameters can optionally be provided to select a sub-rectangle
    of the output. In this case, only the requested regions
    will be rendered. (Default: Unused)
* - high_quality_edges
  - |bool|
  - If set to |true|, regions slightly outside of the film plane will also be sampled. This may
    improve the image quality at the edges, especially when using very large reconstruction
    filters. In general, this is not needed though. (Default: |false|, i.e. disabled)
* - rfilter
  - :paramtype:`rfilter`
  - Reconstruction filter that should be used by the film in the spatial dimension. (Default: :monosp:`gaussian`, a windowed
    Gaussian filter)
* - tfilter
  - :paramtype:`rfilter`
  - (NOT IMPLEMENTED; right now this is done in the postprocessing step in Python) Reconstruction filter that should be used by the film in the temporal dimension. (Default: :monosp:`gaussian`, a windowed
    Gaussian filter)
 */

template <typename Float, typename Spectrum>
class StreakHDRFilm final : public StreakFilm<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(StreakFilm, m_size, m_crop_size, m_crop_offset, m_num_bins,
                    m_bin_width_opl, m_start_opl, m_auto_detect_bins,
                    m_high_quality_edges, m_filter, m_time_filter)
    MTS_IMPORT_TYPES(ReconstructionFilter, StreakImageBlock, ImageBlock)

    StreakHDRFilm(const Properties &props) : Base(props) {
        std::string file_format = string::to_lower(
            props.string("file_format", "openexr"));
        std::string pixel_format = string::to_lower(
            props.string("pixel_format", "rgba"));
        std::string component_format = string::to_lower(
            props.string("component_format", "float16"));

        m_dest_file = props.string("filename", "");

        if (file_format == "openexr" || file_format == "exr")
            m_file_format = Bitmap::FileFormat::OpenEXR;
        else if (file_format == "rgbe")
            m_file_format = Bitmap::FileFormat::RGBE;
        else if (file_format == "pfm")
            m_file_format = Bitmap::FileFormat::PFM;
        else {
            Throw("The \"file_format\" parameter must either be "
                  "equal to \"openexr\", \"pfm\", or \"rgbe\","
                  " found %s instead.", file_format);
        }

        if (pixel_format == "luminance" || is_monochromatic_v<Spectrum>) {
            m_pixel_format = Bitmap::PixelFormat::Y;
            if (pixel_format != "luminance")
                Log(Warn,
                    "Monochrome mode enabled, setting film output pixel format "
                    "to 'luminance' (was %s).",
                    pixel_format);
        } else if (pixel_format == "luminance_alpha")
            m_pixel_format = Bitmap::PixelFormat::YA;
        else if (pixel_format == "rgb")
            m_pixel_format = Bitmap::PixelFormat::RGB;
        else if (pixel_format == "rgba")
            m_pixel_format = Bitmap::PixelFormat::RGBA;
        else if (pixel_format == "xyz")
            m_pixel_format = Bitmap::PixelFormat::XYZ;
        else if (pixel_format == "xyza")
            m_pixel_format = Bitmap::PixelFormat::XYZA;
        else {
            Throw("The \"pixel_format\" parameter must either be equal to "
                  "\"luminance\", \"luminance_alpha\", \"rgb\", \"rgba\", "
                  " \"xyz\", \"xyza\". Found %s.",
                  pixel_format);
        }

        if (component_format == "float16")
            m_component_format = Struct::Type::Float16;
        else if (component_format == "float32")
            m_component_format = Struct::Type::Float32;
        else if (component_format == "uint32")
            m_component_format = Struct::Type::UInt32;
        else {
            Throw("The \"component_format\" parameter must either be "
                  "equal to \"float16\", \"float32\", or \"uint32\"."
                  " Found %s instead.", component_format);
        }

        if (m_file_format == Bitmap::FileFormat::RGBE) {
            if (m_pixel_format != Bitmap::PixelFormat::RGB) {
                Log(Warn, "The RGBE format only supports pixel_format=\"rgb\"."
                           " Overriding..");
                m_pixel_format = Bitmap::PixelFormat::RGB;
            }
            if (m_component_format != Struct::Type::Float32) {
                Log(Warn, "The RGBE format only supports "
                           "component_format=\"float32\". Overriding..");
                m_component_format = Struct::Type::Float32;
            }
        } else if (m_file_format == Bitmap::FileFormat::PFM) {
            // PFM output; override pixel & component format if necessary
            if (m_pixel_format != Bitmap::PixelFormat::RGB && m_pixel_format != Bitmap::PixelFormat::Y) {
                Log(Warn, "The PFM format only supports pixel_format=\"rgb\""
                           " or \"luminance\". Overriding (setting to \"rgb\")..");
                m_pixel_format = Bitmap::PixelFormat::RGB;
            }
            if (m_component_format != Struct::Type::Float32) {
                Log(Warn, "The PFM format only supports"
                           " component_format=\"float32\". Overriding..");
                m_component_format = Struct::Type::Float32;
            }
        }

        props.mark_queried("banner"); // no banner in Mitsuba 2
    }

    void prepare(const std::vector<std::string> &channels) override {
        std::vector<std::string> channels_sorted = channels;
        channels_sorted.push_back("R");
        channels_sorted.push_back("G");
        channels_sorted.push_back("B");
        std::sort(channels_sorted.begin(), channels_sorted.end());
        for (size_t i = 1; i < channels.size(); ++i) {
            if (channels[i] == channels[i - 1])
                Throw("Film::prepare(): duplicate channel name \"%s\"", channels[i]);
        }

        m_storage = new StreakImageBlock(m_crop_size,
                                         m_num_bins,
                                         m_bin_width_opl,
                                         m_start_opl,
                                         channels.size());
        m_storage->set_offset(m_crop_offset);
        m_storage->clear();
        m_channels = channels;
    }

    void put(const ImageBlock * /* block */) override {
        NotImplementedError("put");
    }

    void put(const StreakImageBlock *block) override {
        Assert(m_storage != nullptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage->put(block);
    }

    bool develop(const ScalarPoint2i  &source_offset,
                 const ScalarVector2i &size,
                 const ScalarPoint2i  &target_offset,
                 Bitmap *target) const override {
        Assert(m_storage != nullptr);
        (void) source_offset;
        (void) size;
        (void) target_offset;
        (void) target;

#if 0
        const Bitmap *source = m_storage->bitmap();

        StructConverter cvt(source->struct_(), target->struct_());

        size_t source_bpp = source->bytes_per_pixel();
        size_t target_bpp = target->bytes_per_pixel();

        const uint8_t *source_data = source->uint8_data()
            + (source_offset.x() + source_offset.y() * source->width()) * source_bpp;
        uint8_t *target_data = target->uint8_data()
            + (target_offset.x() + target_offset.y() * target->width()) * target_bpp;

        if (size.x() == m_crop_size.x() && target->width() == m_storage->width()) {
            // Develop a connected part of the underlying buffer
            cvt.convert(size.x() * size.y(), source_data, target_data);
        } else {
            // Develop a rectangular subregion
            for (int i = 0; i < size.y(); ++i) {
                cvt.convert(size.x(), source_data, target_data);
                source_data += source->width() * source_bpp;
                target_data += target->width() * target_bpp;
            }
        }
#endif
        return true;
    }

    ref<Bitmap> bitmap(bool raw) override {
        if constexpr (is_cuda_array_v<Float>) {
            cuda_eval();
            cuda_sync();
        }

        ref<Bitmap> source = new Bitmap(
            m_channels.size() != 4 ? Bitmap::PixelFormat::MultiChannel : Bitmap::PixelFormat::XYZA,
            struct_type_v<ScalarFloat>,
            {m_storage->size().x() * m_storage->time(), m_storage->size().y()},
            m_storage->channel_count(),
            (uint8_t *) m_storage->data().managed().data()
            );

        if (raw)
            return source;

        bool has_aovs = m_channels.size() != 4;

        ref<Bitmap> target = new Bitmap(
            has_aovs ? Bitmap::PixelFormat::MultiChannel : m_pixel_format,
            m_component_format,
            {m_storage->size().x() * m_storage->time(), m_storage->size().y()},
            has_aovs ? (m_storage->channel_count() - 1) : 0);

        if (has_aovs) {
            for (size_t i = 0, j = 0; i < m_channels.size(); ++i, ++j) {
                Struct::Field &source_field = source->struct_()->operator[](i),
                              &dest_field   = target->struct_()->operator[](j);

                switch (i) {
                    case 0:
                        dest_field.name = "R";
                        dest_field.blend = {
                            {  3.240479f, "X" },
                            { -1.537150f, "Y" },
                            { -0.498535f, "Z" }
                        };
                        break;

                    case 1:
                        dest_field.name = "G";
                        dest_field.blend = {
                            { -0.969256, "X" },
                            {  1.875991, "Y" },
                            {  0.041556, "Z" }
                        };
                        break;

                    case 2:
                        dest_field.name = "B";
                        dest_field.blend = {
                            {  0.055648, "X" },
                            { -0.204043, "Y" },
                            {  1.057311, "Z" }
                        };
                        break;

                    case 4:
                        source_field.flags |= +Struct::Flags::Weight;
                        j--;
                        break;

                    default:
                        dest_field.name = m_channels[i];
                        break;
                }

                source_field.name = m_channels[i];
            }
        }

        source->convert(target);

        return target;
     };

    ref<Bitmap> bitmap(int slice, bool raw) override {
        if constexpr (is_cuda_array_v<Float>) {
            cuda_eval();
            cuda_sync();
        }

        DynamicBuffer<Float> dslice = m_storage->data(slice);

        bool has_aovs = m_channels.size() != 3 && m_channels.size() != 5;

        ref<Bitmap> source = new Bitmap(
            has_aovs ? Bitmap::PixelFormat::MultiChannel
                     : (m_channels.size() == 3 ? Bitmap::PixelFormat::XYZ
                                               : Bitmap::PixelFormat::XYZAW),
            struct_type_v<ScalarFloat>,
            {m_storage->time(), m_storage->width()},
            m_storage->channel_count(),
            (uint8_t *) dslice.managed().data()
            // This second option should work too (similar to hdrfilm.cpp), but I do not why,
            // it makes the first 4 values of the slice to be corrupt/incorrect. The upper way
            // doing it in two steps with the variable dslice, makes it work. It probably has
            // to do with some innerworkings of Enoki and how m_storage->data(int slice)
            // returns a DynamicBuffer as a copy and not as a reference to m_storage.
            //--> (uint8_t *) m_storage->data(slice).managed().data()
            );

        if (raw)
            return source;

        ref<Bitmap> target = new Bitmap(
            has_aovs ? Bitmap::PixelFormat::MultiChannel : m_pixel_format,
            m_component_format,
            {m_storage->time(), m_storage->width()},
            // NOTE(diego): old code had "channel_count() - 1" here,
            // but it makes scalar_rgb_polarized not work.
            has_aovs ? m_storage->channel_count() : 0
            );

// NOTE (diego): this part makes scalar_rgb_polarized not work
        if (has_aovs) {
            for (size_t i = 0, j = 0; i < m_channels.size(); ++i, ++j) {
                Struct::Field &source_field = source->struct_()->operator[](i),
                    &dest_field   = target->struct_()->operator[](j);

#if 0
                switch (i) {
                    case 0:
                        dest_field.name = "R";
                        dest_field.blend = {
                            {  3.240479f, "X" },
                            { -1.537150f, "Y" },
                            { -0.498535f, "Z" }
                        };
                        break;

                    case 1:
                        dest_field.name = "G";
                        dest_field.blend = {
                            { -0.969256, "X" },
                            {  1.875991, "Y" },
                            {  0.041556, "Z" }
                        };
                        break;

                    case 2:
                        dest_field.name = "B";
                        dest_field.blend = {
                            {  0.055648, "X" },
                            { -0.204043, "Y" },
                            {  1.057311, "Z" }
                        };
                        break;

                    case 4:
                        source_field.flags |= +Struct::Flags::Weight;
                        j--;
                        break;

                    default:
                        dest_field.name = m_channels[i];
                        break;
                }
#else
                dest_field.name = m_channels[i];
#endif

                source_field.name = m_channels[i];
            }
        }

        source->convert(target);

        return target;
    };

    void develop() override {
        //TODO: improve the way this handles the creation of the file destination
        if (m_dest_file.empty())
            Throw("Destination file not specified, cannot develop.");

        fs::path directoryname = m_dest_file;
        std::string proper_extension;
        if (m_file_format == Bitmap::FileFormat::OpenEXR)
            proper_extension = ".exr";
        else if (m_file_format == Bitmap::FileFormat::RGBE)
            proper_extension = ".rgbe";
        else
            proper_extension = ".pfm";

        // Remove extension if it exists
        fs::create_directory(directoryname.replace_extension());

        for(int i = 0; i < m_size.y(); ++i) {
            std::string filename_str = "frame_" + std::to_string(i);
            fs::path filename = fs::path(filename_str);
            std::string extension = string::to_lower(filename.extension().string());
            if (extension != proper_extension)
                filename.replace_extension(proper_extension);

            filename = directoryname / filename;
            Log(Info, "\U00002714  Developing \"%s\" ..", filename.string());

            bitmap(i, false)->write(filename, m_file_format);
        }
    }

    /**
     * This method writes the streakimage as a single image of dimension [width x time, height]
    void develop() override {

        if (m_dest_file.empty())
            Throw("Destination file not specified, cannot develop.");

        fs::path filename = m_dest_file;
        std::string proper_extension;
        if (m_file_format == Bitmap::FileFormat::OpenEXR)
            proper_extension = ".exr";
        else if (m_file_format == Bitmap::FileFormat::RGBE)
            proper_extension = ".rgbe";
        else
            proper_extension = ".pfm";

        std::string extension = string::to_lower(filename.extension().string());
        if (extension != proper_extension)
            filename.replace_extension(proper_extension);

        Log(Info, "\U00002714  Developing \"%s\" ..", filename.string());

        bitmap(false)->write(filename, m_file_format);
    }
     **/

    void set_destination_file(const fs::path &dest_file) override {
        m_dest_file = dest_file;
    }

    bool destination_exists(const fs::path &base_name) const override {
        std::string proper_extension;
        if (m_file_format == Bitmap::FileFormat::OpenEXR)
            proper_extension = ".exr";
        else if (m_file_format == Bitmap::FileFormat::RGBE)
            proper_extension = ".rgbe";
        else
            proper_extension = ".pfm";

        fs::path filename = base_name;

        std::string extension = string::to_lower(filename.extension().string());
        if (extension != proper_extension)
            filename.replace_extension(proper_extension);

        return fs::exists(filename);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "StreakHDRFilm[" << std::endl
            << "  size = " << m_size << "," << std::endl
            << "  crop_size = " << m_crop_size << "," << std::endl
            << "  crop_offset = " << m_crop_offset << "," << std::endl
            << "  num_bins = " << m_num_bins << "," << std::endl
            << "  bin_width_opl = " << m_bin_width_opl << "," << std::endl
            << "  start_opl = " << m_start_opl << "," << std::endl
            << "  high_quality_edges = " << m_high_quality_edges << "," << std::endl
            << "  filter = " << m_filter << std::endl
            << "  time_filter = " << m_time_filter << std::endl
            << "  file_format = " << m_file_format << "," << std::endl
            << "  pixel_format = " << m_pixel_format << "," << std::endl
            << "  component_format = " << m_component_format << "," << std::endl
            << "  dest_file = \"" << m_dest_file << "\"" << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    Bitmap::FileFormat m_file_format;
    Bitmap::PixelFormat m_pixel_format;
    Struct::Type m_component_format;
    fs::path m_dest_file;
    ref<StreakImageBlock> m_storage;
    std::mutex m_mutex;
    std::vector<std::string> m_channels;
};

MTS_IMPLEMENT_CLASS_VARIANT(StreakHDRFilm, StreakFilm)
MTS_EXPORT_PLUGIN(StreakHDRFilm, "Streak HDR Film")
NAMESPACE_END(mitsuba)
