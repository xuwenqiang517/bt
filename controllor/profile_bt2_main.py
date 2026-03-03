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
    print("=" * 80)
    print("Profiling results (top 30 by cumulative time, excluding stock_data.py and bt2.py):")
    print("=" * 80)
    stats = pstats.Stats(profile_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    # Filter out functions from stock_data.py and bt2.py, and only show functions with cumulative time > 0.1 seconds
    stats.print_stats(30, lambda func: 'stock_data.py' not in func and 'bt2.py' not in func)

    # Also print by tottime to see where the actual time is spent
    print("\n" + "=" * 80)
    print("Profiling results (top 30 by total time, excluding stock_data.py and bt2.py):")
    print("=" * 80)
    stats.sort_stats('tottime')
    # Filter out functions from stock_data.py and bt2.py, and only show functions with total time > 0.1 seconds
    stats.print_stats(30, lambda func: 'stock_data.py' not in func and 'bt2.py' not in func)

    # 查看调用次数最多的函数
    print("\n" + "=" * 80)
    print("Profiling results (top 30 by call count):")
    print("=" * 80)
    stats.sort_stats('ncalls')
    stats.print_stats(30)

    # 查看策略相关函数的详细性能
    print("\n" + "=" * 80)
    print("Strategy-related functions (cumulative time):")
    print("=" * 80)
    stats.sort_stats('cumulative')
    stats.print_stats('strategy.py', 30)

    # 查看chain相关函数的详细性能
    print("\n" + "=" * 80)
    print("Chain-related functions (cumulative time):")
    print("=" * 80)
    stats.sort_stats('cumulative')
    stats.print_stats('chain.py', 30)

if __name__ == "__main__":
    main()
