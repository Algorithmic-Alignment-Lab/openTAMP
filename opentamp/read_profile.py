import pstats
from pstats import SortKey
p = pstats.Stats('profile_out')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(50)
