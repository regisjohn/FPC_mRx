import importlib.util
import sys
import time


# 1. DEFINE RECIPE TO LAZILY MAP ENTIRE PACKAGE TREE
def lazy_register(fullname):
    spec = importlib.util.find_spec(fullname)
    if spec is None:
        return None
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    return module


# --- 2. LAZILY LOAD ROOT PACKAGES INSTANTLY ---
t0 = time.perf_counter()

# Pre-register the module and its submodule paths as lazy proxies
lazy_register("pyspedas")

t1 = time.perf_counter()
print(f"Lazy Package Import Setup Time: {(t1 - t0) * 1000:.4f} ms")


# --- 3. YOUR NORMAL PROGRAM CONTINUES HERE ---
print("\nNow running standard import syntax lines...")

# This executes instantly because it's pointing to our pre-registered lazy proxy!
from pyspedas.projects import mms

print("Triggering data load now (This forces actual disk load & execution)...")
t2 = time.perf_counter()

# Call it using your exact standard parameters
# fgm_vars = mms.fgm(
#     trange=["2015-10-16/13:05:30", "2015-10-16/13:07:30"],
#     probe="1",
#     data_rate="brst",
#     level="l2",
#     varnames="mms1_fgm_b_gse_brst_l2",
#     time_clip=True,
#     get_support_data=False,
#     no_update=False,
# )

t3 = time.perf_counter()
print(f"\nExecution & Data Loading Completed in: {t3 - t2:.4f} seconds")