import cProfile
import pstats
import sys
from stock_calendar import StockCalendar as sc
import bt2

def main():
    # Get day_array and result_file exactly as in the main section of bt2.py
    day_array = sc().get_date_arr()
    result_file = f"连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-vol_rank正排"
    
    # Run profiling on the actual bt_all function
    profile_file = "bt2_main_profile.out"
    cProfile.runctx('bt2.bt_all(day_array, result_file)', globals(), locals(), profile_file)
    
    # Print statistics
    print("Profiling results (top 20 by cumulative time):")
    stats = pstats.Stats(profile_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    # Also print by tottime to see where the actual time is spent
    print("\nProfiling results (top 20 by total time):")
    stats.sort_stats('tottime')
    stats.print_stats(20)

if __name__ == "__main__":
    main()
