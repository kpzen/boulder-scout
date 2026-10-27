import os

# OpenBLAS's thread pool costs more than it saves on the DTM gap fill's millions of tiny LAPACK
# calls: 230 s -> 25 s on a 2.5 km tile, same results. Must be set before numpy loads;
# the DTM pool's workers inherit it either way.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
