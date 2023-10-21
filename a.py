# import threading
# import time
# import psutil

# # Function to simulate CPU stress
# def cpu_stress():
#     while True:
#         pass

# # Function to simulate memory stress
# def memory_stress():
#     memory_list = []
#     while True:
#         memory_list.append(' ' * 1024)

# # Function to simulate throughput stress (network)
# def throughput_stress():
#     while True:
#         # You can simulate network activity or data transfer here
#         pass

# # Create threads for each stress test
# cpu_thread = threading.Thread(target=cpu_stress)
# memory_thread = threading.Thread(target=memory_stress)
# throughput_thread = threading.Thread(target=throughput_stress)

# # Start the threads
# cpu_thread.start()
# memory_thread.start()
# throughput_thread.start()

# # Record memory usage every second for 60 seconds
# for _ in range(3):
#     memory_info = psutil.virtual_memory()
#     print(f"Memory Usage: {memory_info.percent}%")
#     time.sleep(1)

# # Stop the threads
# cpu_thread.join()
# memory_thread.join()
# throughput_thread.join()

import stressinjector as injector
# injector.MemoryStress(gigabytes=1000)
injector.CPUStress(seconds=100)