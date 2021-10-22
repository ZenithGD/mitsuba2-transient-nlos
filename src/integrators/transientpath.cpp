#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/transientintegrator.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class TransientPathIntegrator : public TransientMonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(TransientMonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    TransientPathIntegrator(const Properties &props) : Base(props) {
        m_nlos_emitter_sampling = props.bool_("nlos_emitter_sampling", false);
    }

    void sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray_,
                const Medium * /* medium */,
                std::vector<FloatTimeSample<Float, Mask>> & /* aovs_record */,
                std::vector<RadianceSample<Float, Spectrum, Mask>> &timed_samples_record,
                Float max_path_opl,
                Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        RayDifferential3f ray = ray_;

        // Index of refraction of the current medium (used for OPL calculation)
        Float current_ior(lookup_ior("air"));

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f);

        // Time associated to a path, measured in optical path length
        // NOTE(diego): this assumes that the ray's `time` variable is measured
        // in OPL, we'll just have to believe now :-)
        Float path_opl(ray.time);

        Point3f nlos_laser_target;
        if (m_nlos_emitter_sampling) {
            auto emitters = scene->emitters();
            if (unlikely(emitters.size() != 1)) {
                Throw("NLOS emitter sampling is not implemented for scenes "
                      "with more than one emitter.");
            }
            Transform4f trafo =
                emitters[0]->world_transform()->eval(ray.time, true);
            Vector3f laser_dir = trafo.transform_affine(Vector3f(0, 0, 1));
            Ray3f ray_laser(trafo.translation(), laser_dir, ray.time);
            nlos_laser_target = scene->ray_intersect(ray_laser, true).p;
        }

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        EmitterPtr emitter = si.emitter(scene);

        for (int depth = 1;; ++depth) {

            // FIXME(diego): medium index of refraction should be used to scale
            // the distance in order to obtain OPL.
            path_opl += si.distance(ray) * current_ior;

            // Transient samples are not going to get stored anyway, discard
            if (path_opl > max_path_opl)
                break;

            // ---------------- Intersection with emitters ----------------

            if (any_or<true>(neq(emitter, nullptr))) {
                Spectrum radiance(0.f);
                radiance[active] += emission_weight * throughput * emitter->eval(si, active);
                Mask path_finished = active;
                timed_samples_record.emplace_back(path_opl, radiance, path_finished);
            }

            active &= si.is_valid();

            // Stop if we've exceeded the number of requested bounces, or
            // if there are no more active lanes. Only do this latter check
            // in GPU mode when the number of requested bounces is infinite
            // since it causes a costly synchronization.
            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;

            // --------------------- Emitter sampling ---------------------

            BSDFContext ctx;
            BSDFPtr bsdf  = si.bsdf(ray);
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (likely(any_or<true>(active_e))) {
                const auto f_emitter_sample =
                    [*this](const Scene *scene, Sampler *sampler,
                            BSDFContext &ctx, SurfaceInteraction3f &si,
                            Mask &active_e,
                            std::vector<RadianceSample<Float, Spectrum, Mask>>
                                &timed_samples_record,
                            const BSDFPtr &bsdf, const Spectrum &throughput,
                            const Float &path_opl, const Float &current_ior) {
                        auto [ds, emitter_val] =
                            scene->sample_emitter_direction(
                                si, sampler->next_2d(active_e), true, active_e);
                        active_e &= neq(ds.pdf, 0.f);

                        // Query the BSDF for that emitter-sampled direction
                        Vector3f wo       = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                        // Determine density of sampling that same direction
                        // using BSDF sampling
                        Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

                        Float mis =
                            select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                        Spectrum radiance(0.f);
                        radiance[active_e] +=
                            mis * throughput * bsdf_val * emitter_val;

                        timed_samples_record.emplace_back(
                            path_opl + ds.dist * current_ior, radiance,
                            active_e);
                    };

                if (!m_nlos_emitter_sampling) {
                    // same emitter sampling as before
                    f_emitter_sample(scene, sampler, ctx, si, active_e,
                                     timed_samples_record, bsdf, throughput,
                                     path_opl, current_ior);
                } else {
                    // nlos scenes only have one laser emitter - standard
                    // emitter sampling techniques do not apply as most
                    // directions do not emit any radiance, it needs to be very
                    // lucky to bsdf sample the exact point that the laser is
                    // illuminating
                    //
                    // this modifies the emitter sampling so instead of directly
                    // sampling the laser we sample (1) the point that the laser
                    // is illuminating and then (2) the laser

                    // 1. Obtain direction to NLOS illuminated point
                    //    and test visibility with ray_test
                    Vector3f d = nlos_laser_target - si.p;
                    Float dist = norm(d);
                    d /= dist;
                    Ray3f ray_bsdf(si.p, d,
                                   math::RayEpsilon<Float> *
                                       (1.f + hmax(abs(si.p))),
                                   dist * (1.f - math::ShadowEpsilon<Float>),
                                   si.time, si.wavelengths);
                    // active rays are those that did NOT intersect
                    active_e &= !scene->ray_test(ray_bsdf, active_e);

                    // 2. Evaluate BSDF to desired direction
                    if (d.z() < -math::RayEpsilon<Float> &&
                        any_or<true>(active_e)) {
                        Vector3f wo       = si.to_local(d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                        ray_bsdf.maxt = math::Infinity<Float>;
                        SurfaceInteraction3f si_bsdf =
                            scene->ray_intersect(ray_bsdf, active_e);
                        BSDFPtr bsdf_next = si_bsdf.bsdf(ray_bsdf);

                        // 3. Combine nlos + emitter sampling
                        f_emitter_sample(scene, sampler, ctx, si_bsdf, active_e,
                                         timed_samples_record, bsdf_next,
                                         throughput * bsdf_val,
                                         path_opl + dist * current_ior,
                                         current_ior);
                    }
                }
            }

            // ----------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] =
                bsdf->sample(ctx, si, sampler->next_1d(active), sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            current_ior = bs.eta;

            // Intersect the BSDF ray against the scene geometry
            ray                          = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

            /* Determine probability of having sampled that same
               direction using emitter sampling. */
            emitter = si_bsdf.emitter(scene, active);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds), 0.f);

                emission_weight = mis_weight(bs.pdf, emitter_pdf);
            }

            si = std::move(si_bsdf);
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("TransientPathIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
private:
    bool m_nlos_emitter_sampling;
};

MTS_IMPLEMENT_CLASS_VARIANT(TransientPathIntegrator, TransientMonteCarloIntegrator)
MTS_EXPORT_PLUGIN(TransientPathIntegrator, "Transient Path Tracer integrator");
NAMESPACE_END(mitsuba)
