# Change Log
All notable changes to this project will be documented in this file.

This changelog is to adhere to the format given at [keepachangelog](keepachangelog.com/en/1.0.0/) 
and should maintain [semantic versioning](semver.org).

All text added must be human readable. 

Copy and pasting the git commit messages is __NOT__ enough.

## [Unrealeased]

## [0.4.15] - 2021-03-18
### Added
- This CHANGELOG as a change log to help keep track of changes in the SMARTS project that can get easily lost.
- Hosted Documentation on `readthedocs` and pointed to the smarts paper and useful parts of the documentation in the README.
- Running imitation learning will now create a cached history_mission.pkl file in scenario folder that stores 
the missions for all agents.
- Added ijson as a dependency. 
- Added cached_property as a dependency.
### Changed
- Lowered CPU cost of waypoint generation. This will result in a small increase in memory usage.
- Set the number of processes used in `make test` to ignore 2 CPUs if possible.
- Use the dummy OpEn agent (open-agent version 0.0.0) for all examples.
- Improved performance by removing unused traffic light functionality.
- Limit the memory use of traffic histories by incrementally loading the traffic history file with a worker process.
### Fixed
- In order to avoid precision issues in our coordinates with big floating point numbers,
we now initially shift road networks (maps) that are offset back to the origin
using [netconvert](https://sumo.dlr.de/docs/netconvert.html).
We adapt Sumo vehicle positions to take this into account to allow Sumo to continue
using the original coordinate system.  See Issue #325.
- Cleanly close down the traffic history provider thread. See PR #665.
- Improved the disposal of a SMARTS instance. See issue #378.
- Envision now resumes from current frame after un-pausing.
- Skipped generation of cut-in waypoints if they are further off-road than SMARTS currently supports to avoid process crash.
- Fix envision error 15 by cleanly shutting down the envision worker process.

## [Format] - 2021-03-12
### Added 
– Describe any new features that have been added since the last version was released.
### Changed 
– Note any changes to the software’s existing functionality.
### Deprecated
– Note any features that were once stable but are no longer and have thus been removed.
### Fixed
– List any bugs or errors that have been fixed in a change.
### Removed
– Note any features that have been deleted and removed from the software.
### Security
– Invite users to upgrade and avoid fixed software vulnerabilities.
