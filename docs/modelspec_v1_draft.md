# ModelSpec v1 Draft

This is the draft JSON contract for C++ Stage 1. Python should adopt it in Stage 2.

## Channel roles

```text
matter          Transported by Flow mode.
hidden_reserved Reserved for future carried hidden/rule-code channels.
static_env      Reserved for future environment fields, not transported.
render_only     Not used by update.
```

Stage 1 should only implement `matter` robustly, but the parser should preserve unknown/future roles where possible.

## Update modes

```text
expanded_additive
flow
```

`expanded_additive` uses hard clamp by default. `flow` uses mass-conservative transport and no additive clamp on matter, though values should remain finite/non-negative.

## Kernel families

```text
legacy_shell
smooth_gaussian_mixture
```

`legacy_shell` exists for Lenia3D conversion/debug. `smooth_gaussian_mixture` is the new differentiable-oriented default.

## Growth families

```text
polynomial_lenia3d
gaussian
```

Avoid exposing step growth in new presets unless explicitly needed for legacy tests.

## Example

```json
{
  "format_version": 1,
  "model_type": "expanded_flow",
  "state": {
    "channels": [
      {"name": "body", "role": "matter"}
    ],
    "render_channel": 0
  },
  "update": {
    "mode": "flow",
    "dt": 0.2,
    "flow": {
      "theta_A": 1.0,
      "alpha_power": 2.0,
      "flow_max": 1.0,
      "transport_sigma": 0.5,
      "border": "torus",
      "gradient": "sobel3d"
    }
  },
  "kernels": [
    {
      "name": "body_self_0",
      "src": 0,
      "dst": 0,
      "weight": 1.0,
      "family": "smooth_gaussian_mixture",
      "R": 12.0,
      "envelope_sharpness": 10.0,
      "basis": [
        {"center": 0.25, "width": 0.08, "amplitude": 1.0},
        {"center": 0.55, "width": 0.12, "amplitude": 0.5}
      ],
      "growth": {"family": "gaussian", "mu": 0.12, "sigma": 0.015}
    }
  ]
}
```
