import psutil
import os
import csv
import time

def initialize_csv():
    file_name = "system.csv"
    header = [
        "cpu_usage",
        "cpu_core_count",
        "cpu_thread_count",
        "disk_utilization",
        "memory_usage_bytes",
        "memory_utilization",
        "percent_disk_reads",
        "percent_disk_writes",
        "percent_disk_read_bytes",
        "percent_disk_write_bytes",
        "average_cpu_utilization_by_threads"
    ]

    if not os.path.isfile(file_name):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

def collect_data():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_core_count = psutil.cpu_count(logical=False)
        cpu_thread_count = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        memory_usage = memory.used
        memory_utilization = memory.percent
        disk_utilization = psutil.disk_usage('/').percent  # Disk utilization as a percentage
        disk_io = psutil.disk_io_counters()
        total_disk_reads = disk_io.read_count
        total_disk_writes = disk_io.write_count
        total_disk_read_bytes = disk_io.read_bytes
        total_disk_write_bytes = disk_io.write_bytes
        time.sleep(5)
        disk_io = psutil.disk_io_counters()
        new_disk_reads = disk_io.read_count
        new_disk_writes = disk_io.write_count
        new_disk_read_bytes = disk_io.read_bytes
        new_disk_write_bytes = disk_io.write_bytes
        read_count_diff = new_disk_reads - total_disk_reads
        write_count_diff = new_disk_writes - total_disk_writes
        read_bytes_diff = new_disk_read_bytes - total_disk_read_bytes
        write_bytes_diff = new_disk_write_bytes - total_disk_write_bytes
        total_disk_operations = read_count_diff + write_count_diff
        total_disk_data = read_bytes_diff + write_bytes_diff
        percent_disk_reads = (read_count_diff / total_disk_operations) * 100 if total_disk_operations else 0
        percent_disk_writes = (write_count_diff / total_disk_operations) * 100 if total_disk_operations else 0
        percent_disk_read_bytes = (read_bytes_diff / total_disk_data) * 100 if total_disk_data else 0
        percent_disk_write_bytes = (write_bytes_diff / total_disk_data) * 100 if total_disk_data else 0
        cpu_utilization = psutil.cpu_percent(interval=1, percpu=True)
        total_cpu_utilization = sum(cpu_utilization)
        cpu_thread_utilization = total_cpu_utilization / len(cpu_utilization)
        data = [
            cpu_usage,
            cpu_core_count,
            cpu_thread_count,
            disk_utilization,
            memory_usage,
            memory_utilization,
            percent_disk_reads,
            percent_disk_writes,
            percent_disk_read_bytes,
            percent_disk_write_bytes,
            cpu_thread_utilization
        ]
        print(data)
        file_name = "system.csv"

        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        time.sleep(3) 

initialize_csv()
collect_data()
