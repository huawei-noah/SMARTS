# Change Log
All notable changes to this project will be documented in this file.

This changelog is to adhere to the format given at [keepachangelog](keepachangelog.com/en/1.0.0/) 
and should maintain [semantic versioning](semver.org).

All text added must be human readable. 

Copy and pasting the git commit messages is __NOT__ enough.

## [Unrealeased]
### Added
- This CHANGELOG as a change log to help keep track of changes in the SMARTS project that can get easily lost.
- Hosted Documentation on readthedocs and pointed to documentations and smarts paper in README
- Running imitation learning will now create a cached history_mission.pkl file in scenario folder that stores 
the missions for all agents.
### Fixed
- In order to avoid precision issues in our coordinates with big floating point numbers,
we now initially shift road networks (maps) that are offset back to the origin
using [netconvert](https://sumo.dlr.de/docs/netconvert.html).
We adapt Sumo vehicle positions to take this into account to allow Sumo to continue
using the original coordinate system.  See Issue #325.
- Refactored SMARTS class to not inherit from Panda3D's ShowBase; it's aggregated instead. 
See Issue #407.

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
