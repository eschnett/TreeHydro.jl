create a visualization (movie) of a kelvin-helmholtz instability that
looks crisp and uses many refinement levels.

as the simulation progresses, new levels should be added. how many
levels can we handle? 10? (i hope so.) 20? (maybe that's too many.) i
am thinking of running on symmetry, using an h200 gpu there. the total
run time should be less than 24 hours, but it might be memory usage
that limits the run.

as more levels are added, the simulation will slow down because (a)
there are more blocks and (b) the time step will decrease. at the same
time, the new levels will contain details that are then too fine to be
seen. i envision the camera zooming in to an interesting location so
that all fine features remain visible, probably using a zoom level
corresponding to the number of refinement levels. the simulation time
in the movie would then also slow down so that one would see the same
feature speed per pixel. (is this true for KH?)

use a high effective resolution at all times. 256^2 might not be
enough, 512^2 might be too much. i want the movie to look crisp e.g.
on youtube or when shown in a presentation.

use reflecting boundary conditions in the y direction to reduce the
simulated domain size in half, then complete the domain for
visualization.

at the end of the movie, when the simulation has finished, the camera
could zoom out again, showing how far it zoomed in during the movie.

i assume that either the simultion or rendering the movie could be
expensive. TreeHydro now supports checkpointing. it may make sense to
run the simulation first, outputting data along the way, and rendering
the movie in a second, independent step. whether this makes sense
depends on which step is more expensive. if the simulation is
expensive then a short exploratory run with a coarser resolution
(taking less than 1 hour) would make sense.
