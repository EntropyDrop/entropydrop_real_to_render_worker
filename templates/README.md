# Production templates

The committed order is:

1. `template41.png` — 图2
2. `template51.png` — 图3
3. `template52.png` — 图4

The backend registry binds filenames and ordering to each model version. The
worker receives that ordered list in the task and resolves every filename under
`TEMPLATES_ROOT_DIR`.
