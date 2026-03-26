# Examples

```{toctree}
:maxdepth: 2

basic_usage
```

## Basic Examples

Here are some basic examples of using GeoidLab. For more detailed walkthroughs, see the [tutorial repository](https://github.com/cikelly/geoidlab-tutorial).

### Gravity Reduction

```bash
geoidlab reduce --input-file gravity.csv --model EGM2008 \
                --do helmert --grid \
                --grid-size 1 --grid-unit minutes \
                --bbox -5 5 -5 5 --grid-method kriging
```

### Residual Geoid Computation

```bash
geoidlab geoid --input-file surface_gravity.csv --model EGM2008 \
               --bbox -5 5 -5 5 --grid-size 5 --grid-unit minutes \
               --method hg --sph-cap 1.0 --max-deg 2190
```

### Tide System Conversion

```bash
geoidlab geoid --target-tide-system tide_free \
               --gravity-tide mean_tide
```

### Survey Table Preparation

```bash
geoidlab prep --input-file raw_gravity.xlsx --output-file surface_gravity.csv \
              --platform terrestrial --data-type gravity --tide-system mean_tide
```

See the individual example page for more detailed explanations and use cases.
