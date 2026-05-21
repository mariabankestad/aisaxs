# Documentation site

This folder is published as a GitHub Pages site at:

> https://mariabankestad.github.io/aisaxs/

It holds a single conceptual walkthrough of the framework, intended for
readers who want the concepts before running code. The README in the
repository root is the practical reference; this site complements it
rather than replaces it.

## Files

- `index.md` — landing page
- `lnp_saxs_walkthrough.md` — narrative walkthrough of the framework
- `figures/` — figures used by the walkthrough and by the repository
  README

## Enabling GitHub Pages

If Pages is not already enabled on this repository, do this once:

1. Open the repository on GitHub.
2. Settings → Pages.
3. Under "Build and deployment", choose:
   - Source: Deploy from a branch
   - Branch: main
   - Folder: /docs
4. Click Save.

GitHub will publish the site at the URL above. First deploy takes a few
minutes; subsequent updates are usually faster.

## Note for maintainers

Keep the walkthrough conceptual (the "why" and the "what"), not
procedural (the "how"). Procedural detail lives in the root README and
in the tutorial notebooks, where it stays close to the code and updates
naturally when defaults change. This way the docs site does not need
maintenance every time a script default shifts.
