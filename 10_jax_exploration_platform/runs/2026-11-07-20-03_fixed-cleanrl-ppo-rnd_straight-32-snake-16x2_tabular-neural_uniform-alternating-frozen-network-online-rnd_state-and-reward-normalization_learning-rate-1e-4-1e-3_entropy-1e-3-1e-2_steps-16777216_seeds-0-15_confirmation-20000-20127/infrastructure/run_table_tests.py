import sys, traceback
sys.path.insert(0, '/p/rlprojects/RND/10_jax_exploration_platform/tests')
import test_fixed_settings_report_table as T
failed = 0
for name in dir(T):
    if name.startswith('test_'):
        try:
            getattr(T, name)(); print('passed', name)
        except Exception:
            failed += 1; print('FAILED', name); traceback.print_exc()
sys.exit(failed)
