import cProfile
import pstats
import sys
import bt2

def main():
    # Run profiling on the actual bt_all function with new parameters
    # Use 1 processor and limit to 500 strategy parameters
    profile_file = "bt2_main_profile.out"
    cProfile.runctx('bt2.bt_all(processor_count=1, fail_count=2, max_strategy_count=500)', globals(), locals(), profile_file)
    
    # Print statistics with filter to exclude stock_data.py and bt2.py operations
    print("Profiling results (top 20 by cumulative time, excluding stock_data.py and bt2.py):")
    stats = pstats.Stats(profile_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    # Filter out functions from stock_data.py and bt2.py, and only show functions with cumulative time > 0.1 seconds
    stats.print_stats(20, lambda func: 'stock_data.py' not in func and 'bt2.py' not in func)
    
    # Also print by tottime to see where the actual time is spent
    print("\nProfiling results (top 20 by total time, excluding stock_data.py and bt2.py):")
    stats.sort_stats('tottime')
    # Filter out functions from stock_data.py and bt2.py, and only show functions with total time > 0.1 seconds
    stats.print_stats(20, lambda func: 'stock_data.py' not in func and 'bt2.py' not in func)

if __name__ == "__main__":
    main()
