# Zero-degree correction modes

## Goal

Keep GeoidLab's existing full physical-potential zero-degree correction as the
default while offering an explicit ICGEM-compatible mode. The choice must be
independent from the existing `icgem` topographic-correction option.

## Background

The current correction applies

\[
\frac{GM_{model}-GM_{ref}}{\gamma R}
- \frac{W_0-U_0}{\gamma}.
\]

This is the desired default (`physical_w0`). ICGEM's optional zero-degree
correction applies only the model/reference mass difference, using
`GM_model * C00`, and does not apply the `W0 - U0` component. The two
conventions can differ by approximately 0.76 m when using GRS80.

## Public interface

Add `zero_degree_mode: str = "physical_w0"` to these public methods:

- `GlobalGeopotentialModel.zero_degree_term()`
- `GlobalGeopotentialModel.geoid()`
- `GlobalGeopotentialModel.height_anomaly()`

Supported values are:

- `"physical_w0"` (default): retain the current full correction.
- `"icgem"`: apply only `(GM_model * C00 - GM_ref) / (gamma * R)`.
- `"none"`: do not add a zero-degree correction.

Invalid values raise `ValueError` with the supported values in the message.

`GM_model` is derived as `self.shc["GM"] * self.shc["Cnm"][0, 0]`. The
implementation validates that the coefficient exists and is finite. This is
normally numerically identical to the existing implementation because ICGEM
static models conventionally have `C00 == 1`.

## CLI and configuration

Add `--zero-degree-mode` to the reference/GGM CLI, with choices
`physical_w0`, `icgem`, and `none`, defaulting to `physical_w0`. Propagate the
value through `GGMSynthesis` to geoid, height-anomaly, and standalone
zero-degree-term computations. Add the matching default and explanatory
comment to `docs/geoidlab.cfg`.

The existing `--icgem` flag remains unchanged: it controls only the optional
topographic contribution in `geoid()`. It must not implicitly change
`zero_degree_mode`.

## Behavior

`geoid()` and `height_anomaly()` synthesize their existing quantity first and
then delegate the correction to `zero_degree_term()` with the selected mode.
For `none`, the original quantity is returned unchanged. The geoid and height
anomaly retain their current choices of radius and normal gravity when a
correction is requested.

## Testing

Add focused unit tests using a minimal in-memory SHC dictionary:

1. `physical_w0` reproduces the current formula, with `GM * C00`.
2. `icgem` contains the GM/C00 contribution and omits `W0 - U0`.
3. The difference between the first two modes equals
   `-(W0 - U0) / gamma`.
4. `none` returns its input unchanged.
5. A non-unity `C00` changes both corrected modes as expected.
6. Invalid modes and missing/non-finite C00 fail clearly.
7. CLI/parser/config propagation preserves the default and passes an explicit
   ICGEM mode independently of the topographic `icgem` flag.

## Compatibility and documentation

The default remains `physical_w0`; existing users retain their numerical
outputs. Documentation and help text must state that `icgem` is intended for
comparison with ICGEM's optional zero-degree checkbox, while `physical_w0`
uses GeoidLab's full potential-level convention.
