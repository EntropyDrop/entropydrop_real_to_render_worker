# Production templates

The current v66 order is:

1. `template41.png` — 图2
2. `template51.png` — 图3
3. `template66.png` — 图4
4. `template67.png` — 图5
5. `template68.png` — 图6

The backend registry binds filenames and ordering to each model version. The
worker receives that ordered list in the task and resolves every filename under
`TEMPLATES_ROOT_DIR`.
