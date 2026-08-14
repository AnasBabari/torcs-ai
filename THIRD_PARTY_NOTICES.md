# Third-Party Notices and Licensing

## 1. Project Licensing
The original Python code, environment adapters, controllers, training pipelines,
evaluation tools, and test suites in this repository are licensed under the
[MIT License](LICENSE).

---

## 2. External TORCS Simulator Notice
TORCS (The Open Racing Car Simulator) is an external third-party software package
and is **not** covered by this repository's MIT License.

- **Primary License**: GNU General Public License (GPL) v2 or later.
- **Original Authors & Contributors**: Eric Espié, Christophe Guionneau, and the TORCS development community.
- **SCR (Simulated Car Racing) Championship Patch**: Created by Matteo Loiacono, Luigi Cardamone, Martin V. Butz, Pier Luca Lanzi, et al.
- **Artwork & Content**: Certain 3D car models, tracks, and textures carry specific licenses:
  - Free Art License (FAL) / GNU Free Documentation License (GFDL) for community assets.
  - Car models `data/cars/models/pw-*` (Patwo Design) and `kc-*` (Kcendra) carry non-commercial / author-specific distribution notices as detailed in their respective readme files within the TORCS distribution.

### Obtaining TORCS
Users and researchers should obtain TORCS from official / verified distributions:
- **Windows Pre-configured SCR Distribution**: Extract to `C:\torcs\torcs` (or specify via `$env:TORCS_HOME` / `--torcs-home`).
- **Official SourceForge**: [http://torcs.sourceforge.net](http://torcs.sourceforge.net)
- **SCR Championship**: [http://cs.unibo.it/projects/torcs/](http://cs.unibo.it/projects/torcs/)

### Local Verification
This repository verifies the local TORCS installation in a strictly read-only manner using:
```powershell
python scripts/torcs_doctor.py --torcs-home C:\torcs\torcs
```
The doctor fingerprints `wtorcs.exe`, the SCR server DLL (`scr_server.dll`), XML configurations, driver slots, and benchmark tracks via SHA-256 digests.
