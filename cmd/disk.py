import psutil

# Get CPU utilization for each CPU separately
cpu_utilization = psutil.cpu_percent(interval=1, percpu=True)

# Sum the CPU utilization across all CPUs
total_cpu_utilization = sum(cpu_utilization)

# Calculate the average CPU utilization per CPU
cpu_thread_utilization = total_cpu_utilization / len(cpu_utilization)

print(f"Average CPU utilization by threads: {cpu_thread_utilization:.2f}%")
