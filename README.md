<div style="margin-right: 15px;"><img align="left" src="https://github.com/mitsuba-renderer/mitsuba2/raw/master/docs/images/logo_plain.png" width="90" height="90"/></div>

# <img src="https://render.githubusercontent.com/render/math?math={\huge \frac{\partial}{\partial t} \text{Mitsuba 2}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math={\color{white}\huge \frac{\partial}{\partial t} \text{Mitsuba 2}}#gh-dark-mode-only">

   * [_Download latest version_](https://github.com/diegoroyo/mitsuba2-transient-nlos/releases) / [_Documentation and usage_](#documentation-and-usage) / [_License and citation_](#license-and-citation)

**Authors:** [Diego Royo](https://diego.contact), 
[Jorge Garcia](https://github.com/jgarciapueyo), [Adolfo Muñoz](http://www.adolfo-munoz.com/), [Adrián Jarabo](http://giga.cps.unizar.es/~ajarabo/) and [the original Mitsuba 2 authors](https://github.com/mitsuba-renderer/mitsuba2).

Mitsuba 2, extended for transient path tracing and non-line-of-sight data capture.

## Installation and requirements

This fork of Mitsuba has only be tested on Linux. Things might break on other OS!

HDF5 is also used as a way of storing data without resorting to image oriented formats, bypassing any restrictions on the precision and range of the values stored on the file. For this reason, you must have HDF5 installed on you machine.

You can install Mitsuba easily with the `install.sh` and `build.sh` scripts. They will install **almost** all the requirements, you'll have to install the following additional requirements in order to build Mitsuba correctly:

- Half precision floating point library: `half.hpp`
    Storing half precision floats saves a lot of space (this is how EXR stores the colour data). You must install the `half.hpp` header found [here](https://sourceforge.net/projects/half/) on your `/usr/local/include` folder.

## Documentation and usage

**See 
[tal python library](https://github.com/diegoroyo/tal), with utilies to execute mitsuba and visualize its results.**

### Transient path tracing

We provide the `transientpath`, `transientstokes` and `streakhdrfilm` plugins. There is an example scene below.

```xml
<scene version="2.0.0">
    <default name="spp" value="256"/>
    <default name="res" value="512"/>
    <default name="max_depth" value="4"/>

    <!-- <integrator type="direct"/> -->
    <integrator type="transientpath">
        <integer name="max_depth" value="$max_depth"/>
    </integrator>

    <bsdf type="diffuse" id="box">
        <rgb name="reflectance" value="0.45, 0.30, 0.90"/>
    </bsdf>

    <bsdf type="diffuse" id="white">
        <rgb name="reflectance" value="0.885809, 0.698859, 0.666422"/>
    </bsdf>

    <bsdf type="diffuse" id="red">
        <rgb name="reflectance" value="0.570068, 0.0430135, 0.0443706"/>
    </bsdf>

    <bsdf type="diffuse" id="green">
        <rgb name="reflectance" value="0.105421, 0.37798, 0.076425"/>
    </bsdf>

    <bsdf type="diffuse" id="light">
        <rgb name="reflectance" value="0.936461, 0.740433, 0.705267"/>
    </bsdf>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_luminaire.obj"/>
        <transform name="to_world">
            <translate x="0" y="-0.5" z="0"/>
        </transform>
        <ref id="light"/>
        <ref id="area-emitter"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_floor.obj"/>
        <ref id="white"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_ceiling.obj"/>
        <ref id="white"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_back.obj"/>
        <ref id="white"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_greenwall.obj"/>
        <ref id="green"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_redwall.obj"/>
        <ref id="red"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_smallbox.obj"/>
        <ref id="box"/>
    </shape>
    <shape type="obj">
        <string name="filename" value="meshes/cbox_largebox.obj"/>
        <ref id="box"/>
    </shape>

    <emitter type="area" id="area-emitter">
        <rgb name="radiance" value="18.387, 10.9873, 2.75357"/>
    </emitter>

    <sensor type="perspective">
        <string name="fov_axis" value="smaller"/>
        <float name="near_clip" value="10"/>
        <float name="far_clip" value="2800"/>
        <float name="focus_distance" value="1000"/>
        <float name="fov" value="39.3077"/>
        <transform name="to_world">
            <lookat origin="278, 273, -800"
                    target="278, 273, -799"
                    up    ="  0,   1,    0"/>
        </transform>
        <sampler type="independent">  <!-- ldsampler -->
            <integer name="sample_count" value="$spp"/>
        </sampler>
        <film type="streakhdrfilm" name="streakfilm">
            <string name="file_format" value="hdf5" />
            <integer name="width" value="$res"/>
            <integer name="height" value="$res"/>
            <integer name="time" value="380"/>
            <float name="exposure_time" value="8"/>
            <float name="time_offset" value="520"/>
            <integer name="freq_resolution" value="2" />
            <boolean name="freq_transform" value="true" />
            <float name="lo_fbound" value="-1" />
            <float name="hi_fbound" value="1" />
            <boolean name="high_quality_edges" value="true"/>
            <rfilter name="rfilter" type="box">
                <float name="radius" value="0.3"/>
            </rfilter>
        </film>
    </sensor>
</scene>
```

### Non-line-of-sight data capture

We provide the `nloscapturemeter` plugin. See [nloscapturemeter](https://github.com/diegoroyo/mitsuba2/blob/feat-transient/src/sensors/nloscapturemeter.cpp) for additional details. There is an example scene below.

_Note that variables that start with $ should be changed_

```xml
<scene version="2.2.1">
   <integrator type="transientpath">
      <!-- Recommended 1 for progress bar (see path integrator) -->
      <integer name="block_size" value="1"/>
      <!-- Discard paths with depth >= max_depth -->
      <integer name="max_depth" value="4"/>
      <!-- Only account for paths with depth = filter_depth -->
      <!-- <integer name="filter_depth" value="3"/> -->
      <boolean name="discard_direct_paths" value="true"/>
      <!-- Next event estimation for the laser through the relay wall (recommended true) -->
      <boolean name="nlos_laser_sampling" value="true"/>
      <boolean name="nlos_hidden_geometry_sampling" value="false"/>
      <boolean name="nlos_hidden_geometry_sampling_do_mis" value="false"/>
      <boolean name="nlos_hidden_geometry_sampling_includes_relay_wall" value="false"/>
   </integrator>

   <!-- Relay wall and hidden geometry materials -->
   <bsdf type="diffuse" id="white">
      <rgb name="reflectance" value="0.7, 0.7, 0.7"/>
   </bsdf>

   <!-- Hidden geometry -->
   <shape type="obj">
      <string name="filename" value="$hidden_mesh_obj"/>
      <bsdf type="twosided">
            <ref id="white"/>
      </bsdf>

      <transform name="to_world">
            <scale x="$hidden_scale" y="$hidden_scale" z="$hidden_scale"/>
            <rotate x="1" angle="$hidden_rot_degrees_x"/>
            <rotate y="1" angle="$hidden_rot_degrees_y"/>
            <rotate z="1" angle="$hidden_rot_degrees_z"/>
            <translate x="0" y="0" z="$hidden_distance_to_wall"/>
      </transform>
   </shape>

   <!-- Relay wall -->
   <shape type="rectangle">
      <ref id="white"/>

      <transform name="to_world">
            <rotate z="1" angle="180"/>
            <scale x="$relay_wall_scale" y="$relay_wall_scale" z="1"/>
      </transform>

      <!-- NLOS capture sensor placed on the relay wall -->
      <sensor type="nloscapturemeter">
            <sampler type="independent">
               <integer name="sample_count" value="$sample_count"/>
            </sampler>

            <!-- Laser -->
            <emitter type="projector">
               <spectrum name="irradiance" value="400:0, 500:80, 600:156.0, 700:184.0"/>
               <float name="fov" value="1.5"/>
            </emitter>

            <!-- Acount time of flight for the laser->relay wall and relay wall->sensor paths -->
            <boolean name="account_first_and_last_bounces" value="$account_first_and_last_bounces"/>

            <!-- World-space coordinates -->
            <point name="sensor_origin" x="-0.5" y="0" z="0.25"/>
            <point name="laser_origin" x="-0.5" y="0" z="0.25"/>
            
            <!-- alternative to laser_lookat_pixel -->
            <!-- <point name="laser_lookat_3d" x="0" y="0" z="0"/> -->
            
            <!-- Screen-space coordinates (see streakhdrfilm) -->
            <point name="laser_lookat_pixel" x="$laser_lookat_x" y="$laser_lookat_y" z="0"/>

            <!-- Transient image I(width, height, num_bins) -->
            <film type="streakhdrfilm" name="streakfilm">
               <integer name="width" value="$sensor_width"/>
               <integer name="height" value="$sensor_height"/>

               <!-- Recommended to prevent clamping -->
               <string name="component_format" value="float32"/>

               <integer name="num_bins" value="$num_bins"/>

               <!-- Auto-detect start_opl (and also bin_width_opl if set to a negative value) -->
               <boolean name="auto_detect_bins" value="$auto_detect_bins"/>
               <float name="bin_width_opl" value="$bin_width_opl"/>
               <float name="start_opl" value="$start_opl"/>

               <rfilter name="rfilter" type="box"/>
               <!-- NOTE: tfilters are not implemented yet -->
               <!-- <rfilter name="tfilter" type="box"/>  -->
               <boolean name="high_quality_edges" value="false"/>
            </film>
      </sensor>
   </shape>
</scene>
```

<br>

Refer to the [original Mitsuba 2 repository](https://github.com/mitsuba-renderer/mitsuba2)
for additional documentation and instructions on how to compile, use, and extend Mitsuba 2.

## License and citation

See the [original repository](https://github.com/mitsuba-renderer/mitsuba2). Additionally, if you are using this code in academic research, we would be grateful if you cited [our publication](https://doi.org/10.1016/j.cag.2022.07.003):

```bibtex
@article{royo2022non,
    title = {Non-line-of-sight transient rendering},
    journal = {Computers & Graphics},
    year = {2022},
    issn = {0097-8493},
    doi = {https://doi.org/10.1016/j.cag.2022.07.003},
    url = {https://www.sciencedirect.com/science/article/pii/S0097849322001200},
    author = {Diego Royo and Jorge García and Adolfo Muñoz and Adrian Jarabo}
```
