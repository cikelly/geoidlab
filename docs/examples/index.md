# Examples

```{toctree}
:maxdepth: 2

basic_usage
interpolation
kernel_comparison
tide_conversion
visualization
```

## Basic Examples

Here are some basic examples of using GeoidLab. For more detailed examples, see the [tutorial repository](https://github.com/cikelly/geoidlab-tutorial).

### Gravity Reduction

```bash
geoidlab reduce --input gravity.csv --model EGM2008 \
                --grid-size 1 --grid-unit minutes \
                --bbox [-5,5,-5,5] --grid-method kriging
```

### Residual Geoid Computation

```bash
geoidlab geoid --method hg --sph-cap 1.0 \
               --max-deg 2190 --window-mode cap
```

### Tide System Conversion

```bash
geoidlab geoid --target-tide-system tide_free \
               --gravity-tide mean_tide
```

See the individual example pages for more detailed explanations and use cases.
